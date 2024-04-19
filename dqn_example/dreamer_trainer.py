import torch
import os

from ray.rllib.agents.dreamer import DREAMERTrainer

class CustromDreamerTrainer(DREAMERTrainer):
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = super().save_checkpoint(checkpoint_dir)

        model = self.get_policy().model
        torch.save(model.state_dict(),
                    os.path.join(checkpoint_dir, "checkpoint_state_dict.pth"))

        return checkpoint_path
