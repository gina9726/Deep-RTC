
class step_scheduler(object):

    def __init__(self, optimizer, step_size, gamma):
        self.opt = optimizer
        self.step_size = step_size
        self.decay_rate = gamma
        self.epoch = 1


    def step(self):
        if self.epoch % self.step_size == 0:
            for param_group in self.opt.param_groups:
                param_group['lr'] = param_group['lr'] * self.decay_rate
        self.epoch += 1


    def get_lr(self):
        return self.opt.param_groups[0]['lr']


    def state_dict(self):
        return {'lr': self.get_lr(),
            'epoch': self.epoch,
            'step_size': self.step_size,
            'decay_rate': self.decay_rate
        }


    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.step_size = state_dict['step_size']
        self.decay_rate = state_dict['decay_rate']
        self.step()

