# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings

import torch
from monai.networks.layers.utils import get_act_layer
from monai.utils import LossReduction
from monai.utils.enums import StrEnum
from torch.nn.modules.loss import _Loss
import torch.nn as nn

class RelativisticPatchAdversarialLoss(_Loss):

    def __init__(self, discriminator, gamma=0.001):
        super().__init__()
        self.discriminator = discriminator
        self.gamma = gamma

    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([1, 2, 3])

    def forward(
        self, 
        real_logits: torch.Tensor | list,
        fake_logits: torch.Tensor | list,
        real_samples: torch.Tensor = None,
        fake_samples: torch.Tensor = None,
        for_discriminator: bool = None
    ) -> torch.Tensor | list[torch.Tensor]:
        
        # if type(real_logits) is not list:
        #     real_logits = [real_logits]
        # if type(fake_logits) is not list:
        #     fake_logits = [fake_logits]
            
        # Loss calculation
        loss = []
        
        if for_discriminator == False:
            for r_logit, f_logit in zip(real_logits, fake_logits):
                RelativisticLogits = f_logit - r_logit
                AdversarialLoss = nn.functional.softplus(-RelativisticLogits).mean()
                loss.append(AdversarialLoss)

        if for_discriminator:
            for r_logit, f_logit, r_sample, f_sample in zip(real_logits, fake_logits, real_samples, fake_samples):
                RelativisticLogits = r_logit - f_logit
                AdversarialLoss = nn.functional.softplus(-RelativisticLogits).mean()
                
                r_sample = r_sample.detach().requires_grad_(True).unsqueeze(0)
                f_sample = f_sample.detach().requires_grad_(True).unsqueeze(0)
                
                r_logit = self.discriminator(r_sample)[-1]
                f_logit = self.discriminator(f_sample)[-1]
                
                R1Penalty = self.ZeroCenteredGradientPenalty(r_sample, r_logit)
                R2Penalty = self.ZeroCenteredGradientPenalty(f_sample, f_logit)
                
                DiscriminatorLoss = AdversarialLoss + (self.gamma / 2) * (R1Penalty + R2Penalty)
                loss.append(DiscriminatorLoss)

        return torch.mean(torch.stack(loss))

