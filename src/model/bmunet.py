import timm
import torch.nn
import torchvision
from torch import nn
from torchvision.models import ResNet18_Weights


class MGModule(torch.nn.Module):
    def __init__(self):
        super(MGModule, self).__init__()
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Classification
        self.head = nn.Sequential(
            nn.Linear(resnet.fc.in_features * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 5),
        )

    def forward(self, mg1, mg2):
        # mg1: CC view mg2: MLO view
        mg1_mg2 = torch.cat((mg1, mg2), dim=0)
        mg1_mg2_feature = self.backbone(mg1_mg2)
        mg1_feature, mg2_feature = torch.chunk(mg1_mg2_feature, 2, dim=0)

        mg1_feature_flatten = torch.flatten(mg1_feature, 1, -1)
        mg2_feature_flatten = torch.flatten(mg2_feature, 1, -1)
        # Concat
        total_feature = torch.cat((mg1_feature_flatten, mg2_feature_flatten), dim=1)
        out = self.head(total_feature)
        return out, total_feature, mg1_feature_flatten, mg2_feature_flatten


# Load mirai model as pretrained model
def mg_load_pretrained_model(model: MGModule, mirai_path: str):
    mirai_weight = torch.load(mirai_path, map_location="cpu")
    mirai_weight = mirai_weight.module._model
    model_modules = model.backbone._modules
    model_modules["0"].load_state_dict(mirai_weight.downsampler.conv1.state_dict())
    model_modules["1"].load_state_dict(mirai_weight.downsampler.bn1.state_dict())
    model_modules["2"].load_state_dict(mirai_weight.downsampler.relu.state_dict())
    model_modules["3"].load_state_dict(mirai_weight.downsampler.maxpool.state_dict())

    model_modules["4"]._modules["0"].load_state_dict(mirai_weight.layer1_0.state_dict())
    model_modules["4"]._modules["1"].load_state_dict(mirai_weight.layer1_1.state_dict())
    model_modules["5"]._modules["0"].load_state_dict(mirai_weight.layer2_0.state_dict())
    model_modules["5"]._modules["1"].load_state_dict(mirai_weight.layer2_1.state_dict())
    model_modules["6"]._modules["0"].load_state_dict(mirai_weight.layer3_0.state_dict())
    model_modules["6"]._modules["1"].load_state_dict(mirai_weight.layer3_1.state_dict())
    model_modules["7"]._modules["0"].load_state_dict(mirai_weight.layer4_0.state_dict())
    model_modules["7"]._modules["1"].load_state_dict(mirai_weight.layer4_1.state_dict())
    return model


class MGModuleCAM(MGModule):
    def forward(self, x, **kwargs):
        mg1, mg2 = x
        out, total_feature, _, _ = super().forward(mg1, mg2)
        return out


class USModuleBlock(torch.nn.Module):
    def __init__(self):
        super(USModuleBlock, self).__init__()
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, us_1, us_2):
        us_12 = torch.cat((us_1, us_2), dim=0)
        us_12_feature = self.backbone(us_12)
        us_1_feature, us_2_feature = torch.chunk(us_12_feature, 2, dim=0)

        us_1_feature_flatten = torch.flatten(us_1_feature, 1, -1)
        us_2_feature_flatten = torch.flatten(us_2_feature, 1, -1)
        # Concat
        total_feature = torch.cat((us_1_feature_flatten, us_2_feature_flatten), dim=1)
        return total_feature, us_1_feature_flatten, us_2_feature_flatten


class USModule(nn.Module):
    def __init__(self):
        super(USModule, self).__init__()
        # One model corresponds to one modality
        self.model2 = USModuleBlock()
        self.model3 = USModuleBlock()
        self.model4 = USModuleBlock()
        # Classification
        self.head = nn.Sequential(
            nn.Linear(
                self.model2.get_submodule("backbone.7.1.bn2").num_features * 2 * 3, 2048
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 5),
        )

    def forward(self, us3, us4, us5, us6, us7, us8):
        all_feature1, feature1, feature2 = self.model2(us3, us4)
        all_feature2, feature3, feature4 = self.model3(us5, us6)
        all_feature3, feature5, feature6 = self.model4(us7, us8)
        total_feature = torch.cat((all_feature1, all_feature2, all_feature3), dim=1)
        out = self.head(total_feature)
        return out, total_feature


class USModuleCAM(USModule):
    def forward(self, x, **kwargs):
        us3, us4, us5, us6, us7, us8 = x
        out, total_feature = super().forward(us3, us4, us5, us6, us7, us8)
        return out


class BMUNet(nn.Module):
    def __init__(self, num_clinic_info=10):
        super(BMUNet, self).__init__()
        self.model1 = MGModule()
        self.model2 = USModuleBlock()
        self.model3 = USModuleBlock()
        self.model4 = USModuleBlock()

        self.embed_dim = 512
        self._init_transformer_blocks(num_heads=4, dropout=0.3)
        self._init_position_embeddings()

        self.head = nn.Linear(self.embed_dim + num_clinic_info, 5)

    def _init_transformer_blocks(self, num_heads, dropout):
        ViTBlock = lambda: timm.models.vision_transformer.Block(
            dim=self.embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, proj_drop=dropout, attn_drop=dropout
        )
        self.mg_modal_transformer = ViTBlock()
        self.us_modal_transformer = ViTBlock()
        self.multi_modal_transformer = ViTBlock()
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def _init_position_embeddings(self):
        self.mg_modal_position_embeddings = nn.Parameter(torch.zeros(1, 2, self.embed_dim))
        self.us_modal_position_embeddings = nn.Parameter(torch.zeros(1, 6, self.embed_dim))
        self.multi_modal_position_embeddings = nn.Parameter(
            torch.zeros(1, 2, self.embed_dim))

    def _process_features(self, features, position_embeddings, transformer):
        features += position_embeddings
        features = transformer(features)
        features = self.layer_norm(features)
        pooled_features = torch.mean(features, dim=1)
        return features, pooled_features

    def forward(self, *inputs, clinic_info):
        features = [self.model1(inputs[0], inputs[1])[2:4]] + [model(inputs[i], inputs[i + 1])[1:3] for i, model in
                                                               enumerate([self.model2, self.model3, self.model4],
                                                                         start=2)
                                                               ]

        mg_features = torch.cat([f.unsqueeze(1) for f in features[0]], dim=1)
        us_features = torch.cat([f.unsqueeze(1) for fs in features[1:] for f in fs], dim=1)

        _, mg_pooled = self._process_features(mg_features, self.mg_modal_position_embeddings,
                                              self.mg_modal_transformer)
        _, us_pooled = self._process_features(us_features, self.us_modal_position_embeddings,
                                              self.us_modal_transformer)

        combined_features = torch.cat([mg_pooled.unsqueeze(1), us_pooled.unsqueeze(1)], dim=1)
        _, multi_modal_pooled = self._process_features(combined_features, self.multi_modal_position_embeddings,
                                                       self.multi_modal_transformer)

        combined_info = torch.cat([multi_modal_pooled, clinic_info], dim=1)
        output = self.head(combined_info)

        return output

    def finetune_conv(self):
        for model in [self.model1, self.model2, self.model3, self.model4]:
            for param in model.backbone[:-5].parameters():
                param.requires_grad = True
        self.model1.head.requires_grad_(False)


def bmu_load_pretrained_model(
        model: BMUNet, mg_path: str, us_path: str
):
    mg_weight = torch.load(mg_path, map_location="cpu")
    us_weight = torch.load(us_path, map_location="cpu")
    # Load mg module pretrained weight
    new_mg_weight = {key: value for key, value in mg_weight.items() if not key.startswith("head")}
    model.model1.load_state_dict(new_mg_weight, strict=False)

    # Load us module pretrained weight
    new_us_weights = {"model2": {}, "model3": {}, "model4": {}}
    for key, value in us_weight.items():
        for model_key in new_us_weights.keys():
            if key.startswith(model_key):
                sub_key = key.split(".", 1)[1]
                new_us_weights[model_key][sub_key] = value

    for k, v in new_us_weights.items():
        getattr(model, k).load_state_dict(v, strict=False)
    return model
