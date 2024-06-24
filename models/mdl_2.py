import torch
import torch.nn.functional as F
import torch.nn as nn
import timm
from torch.distributions import Beta
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.cuda.amp import autocast

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)

def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = (norm_att * cla).sum(2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)

class Mixup(nn.Module):
    def __init__(self, mix_beta, mixup_prob, mixup_double):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixup_prob = mixup_prob
        self.mixup_double = mixup_double

    def forward(self, X, Y, weight=None):
        p = torch.rand((1,))[0]
        if p < self.mixup_prob:
            bs = X.shape[0]
            n_dims = len(X.shape)
            perm = torch.randperm(bs)

            p1 = torch.rand((1,))[0]
            if p1 < self.mixup_double:
                X = X + X[perm]
                Y = Y + Y[perm]
                Y = torch.clamp(Y, 0, 1)

                if weight is None:
                    return X, Y
                else:
                    weight = 0.5 * weight + 0.5 * weight[perm]
                    return X, Y, weight
            else:
                perm2 = torch.randperm(bs)
                X = X + X[perm] + X[perm2]
                Y = Y + Y[perm] + Y[perm2]
                Y = torch.clamp(Y, 0, 1)

                if weight is None:
                    return X, Y
                else:
                    weight = (
                        1 / 3 * weight + 1 / 3 * weight[perm] + 1 / 3 * weight[perm2]
                    )
                    return X, Y, weight
        else:
            if weight is None:
                return X, Y
            else:
                return X, Y, weight


class Mixup2(nn.Module):
    def __init__(self, mix_beta, mixup2_prob):
        super(Mixup2, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixup2_prob = mixup2_prob

    def forward(self, X, Y, weight=None):
        p = torch.rand((1,))[0]
        if p < self.mixup2_prob:
            bs = X.shape[0]
            n_dims = len(X.shape)
            perm = torch.randperm(bs)
            coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

            if n_dims == 2:
                X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
            elif n_dims == 3:
                X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
            else:
                X = (
                    coeffs.view(-1, 1, 1, 1) * X
                    + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]
                )
            Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]
            # Y = Y + Y[perm]
            # Y = torch.clamp(Y, 0, 1)

            if weight is None:
                return X, Y
            else:
                weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
                return X, Y, weight
        else:
            if weight is None:
                return X, Y
            else:
                return X, Y, weight

class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg=cfg
        self.bn0 = nn.BatchNorm2d(cfg.mel_spec_args['n_mels'])
        self.num_classes = cfg.n_classes
#         base_model = timm.create_model(
#             cfg.backbone,
#             pretrained=True,
#             in_chans=cfg.in_chans,
#             drop_path_rate=0.2,
#             drop_rate=0.5,
#         )
        # base_model.conv_stem.stride = (1,1)
#         layers = list(base_model.children())[:-2]
#         self.encoder = nn.Sequential(*layers)
        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=cfg.in_chans,
            drop_path_rate=0.2,
            drop_rate=0.5,
        )

        if "efficientnet" in cfg.backbone:
            in_features = self.encoder.num_features
        else:
            in_features = self.encoder.feature_info[-1]["num_chs"]
#         if "efficientnet" in self.cfg.backbone:
#             in_features = base_model.classifier.in_features
#         elif "eca" in self.cfg.backbone:
#             in_features = base_model.head.fc.in_features
#         elif "res" in self.cfg.backbone:
#             in_features = base_model.fc.in_features
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, self.num_classes, activation="sigmoid")

        self.init_weight()

#         self.audio_transforms = Compose(
#             [
#                 # AddColoredNoise(p=0.5),
#                 PitchShift(
#                     min_transpose_semitones=-4,
#                     max_transpose_semitones=4,
#                     sample_rate=self.cfg.SR,
#                     p=0.4,
#                 ),
#                 Shift(min_shift=-0.5, max_shift=0.5, p=0.4),
#             ]
#         )

        self.time_mask_transform = torchaudio.transforms.TimeMasking(
            time_mask_param=60, iid_masks=True, p=0.5
        )
        self.freq_mask_transform = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=24, iid_masks=True
        )

        self.preprocessing = torch.nn.Sequential(MelSpectrogram(**cfg.mel_spec_args),AmplitudeToDB(**cfg.db_args))
        self.norm_by = cfg.norm_by
        
        self.mixup = Mixup(
            mix_beta=self.cfg.mix_beta,
            mixup_prob=self.cfg.mixup_prob,
            mixup_double=self.cfg.mixup_double,
        )
        self.mixup2 = Mixup2(
            mix_beta=self.cfg.mix_beta2, mixup2_prob=self.cfg.mixup2_prob
        )

#         if self.loss == "ce":
#             self.loss_function = nn.CrossEntropyLoss(
#                 label_smoothing=self.cfg.label_smoothing, reduction="none"
#             )
#         elif self.loss == "bce":
#             self.loss_function = nn.BCEWithLogitsLoss(reduction="none")
#         else:
#             raise NotImplementedError
        self.loss_fn0 = nn.BCELoss(reduction="none")
        self.loss_fn1 = nn.BCEWithLogitsLoss(reduction="none")
        
    def transform_to_spec(self, audio):
#         if self.training:
#             audio = self.audio_transforms(audio, sample_rate=self.cfg.SR)


        spec = self.preprocessing(audio)
        spec = (spec + self.norm_by) / self.norm_by

        if self.training:
            spec = self.time_mask_transform(spec)
            if torch.rand(size=(1,))[0] < 0.5:
                spec = self.freq_mask_transform(spec)
#             if torch.rand(size=(1,))[0] < 0.5:
#                 spec = self.lower_upper_freq(spec)
        return spec

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)


    def extract_feature(self,x):
        x = x.permute((0, 1, 3, 2))
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # if self.training:
        #    x = self.spec_augmenter(x)

        x = x.transpose(2, 3)
        # (batch_size, channels, freq, frames)
        x = self.encoder(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        return x, frames_num

    def forward(self, batch):
        x = batch['input']
        y = batch['target']
        weight = batch['weight']
        
        #if test then flatten bs and parts
        if len(x.shape) == 3:
            bs, parts, seq_len = x.shape
            y = torch.repeat_interleave(y[:,None],parts,dim=1)
#             secondary_mask = torch.repeat_interleave(secondary_mask[:,None],parts,dim=1)
            x = x.reshape(bs*parts,seq_len)#.unsqueeze(1)
            n_classes = y.shape[-1]
            y = y.reshape(bs*parts,n_classes)
        x = x[:,None]
#         if not self.training:
#             bs, channel, parts = x.shape[0], x.shape[1], x.shape[2]
#             x = x.reshape((bs * parts, channel, -1))

        if self.training:
            if self.cfg.mixup:
                x, y, weight = self.mixup(x, y, weight)
        with autocast(enabled=False):
            x = self.transform_to_spec(x)
#         if self.in_chans == 3:
#             x = image_delta(x)

        if self.training:
            if self.cfg.mixup2:
                x, y, weight = self.mixup2(x, y, weight)

        x, frames_num = self.extract_feature(x)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

#         interpolate_ratio = frames_num // segmentwise_output.size(1)

#         # Get framewise output
#         framewise_output = interpolate(segmentwise_output, interpolate_ratio)
#         framewise_output = pad_framewise_output(framewise_output, frames_num)

#         framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
#         framewise_logit = pad_framewise_output(framewise_logit, frames_num)
        output_dict = {
#             "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
#             "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output,
        }
#         if not self.training:
#             clipwise_output = clipwise_output.reshape((bs, parts, -1)).max(dim=1).values
#             seg_num = segmentwise_logit.shape[1]
# #             fram_num = framewise_logit.shape[1]
#             segmentwise_logit = (
#                 segmentwise_logit.reshape((bs, parts, seg_num, -1)).max(dim=1).values
#             )
# #             framewise_logit = (
# #                 framewise_logit.reshape((bs, parts, fram_num, -1)).max(dim=1).values
# #             )
        with autocast(enabled=False):
            loss = 0.5 * self.loss_fn0(clipwise_output, y) + 0.5 * self.loss_fn1(segmentwise_logit.max(1)[0], y)
            # loss = 0.5*self.loss_function(torch.logit(clipwise_output), y) + 0.5*self.loss_function(framewise_logit.max(1)[0], y)
    #         if self.loss == "ce":
    #             loss = (loss * weight) / weight.sum()
    #         elif self.loss == "bce":
            if self.training:
                loss = loss.sum(dim=1) * weight
    #         else:
    #             raise NotImplementedError
            loss = loss.sum()


        return {'loss': loss, 'logits': clipwise_output, 'target': y}

