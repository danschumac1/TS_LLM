import os
import time
import torch
from tqdm import tqdm
from loss import MSE
from torch.optim.lr_scheduler import LambdaLR

class Trainer():
    def __init__(self, args, model, train_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = model.to(torch.device(self.device))

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps
        self.weight_decay = args.weight_decay
        self.model_name = self.model.get_name()
        self.print_process(self.model_name)

        self.cr = MSE(self.model)

        self.num_epoch = args.num_epoch
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        if self.num_epoch:
            self.result_file = open(self.save_path + '/result.txt', 'w')
            self.result_file.close()

        self.step = 0
        self.best_metric = -1e9
        self.metric = 'mse'

    def train(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            self.print_process(
                'Basic Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
            print('Basic Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
            self.result_file.close()
        self.print_process(self.best_metric)
        return self.best_metric

    def _to_device(self, batch):
        """Move batch to self.device. Handles Tensor, (tuple/list) of tensors, or dicts."""
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(b) for b in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch  # leave non-tensor types alone

    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader) if self.verbose else self.train_loader

        loss_sum = 0.0
        num_steps = 0

        for idx, batch in enumerate(tqdm_dataloader, start=1):
            batch = self._to_device(batch)  # <<< move to CUDA/CPU

            self.optimizer.zero_grad(set_to_none=True)
            loss = self.cr.compute(batch)
            loss_sum += float(loss.item())
            num_steps += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            self.step += 1
            if self.step % self.lr_decay_steps == 0:
                self.scheduler.step()

            if self.step % self.eval_per_steps == 0:
                metric = self.eval_model_vqvae()
                self.print_process(metric)

                os.makedirs(self.save_path, exist_ok=True)  # ensure dir exists
                with open(os.path.join(self.save_path, "result.txt"), "a+") as f:
                    print(f"step{self.step}", file=f)
                    print(metric, file=f)

                if metric[self.metric] >= self.best_metric:
                    self.model.eval()
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, "model.pkl"))
                    with open(os.path.join(self.save_path, "result.txt"), "a+") as f:
                        print(f"saving model of step{self.step}", file=f)
                    self.best_metric = metric[self.metric]
                    self.model.train()

        avg_loss = loss_sum / max(1, num_steps)
        return avg_loss, time.perf_counter() - t0

    def eval_model_vqvae(self):
        self.model.eval()
        tqdm_data_loader = tqdm(self.test_loader) if self.verbose else self.test_loader

        total_mse = 0.0
        count = 0
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in tqdm_data_loader:
                # move batch to same device as model
                if torch.is_tensor(batch):
                    batch = batch.to(device, non_blocking=True)
                elif isinstance(batch, (list, tuple)):
                    batch = type(batch)(
                        b.to(device, non_blocking=True) if torch.is_tensor(b) else b
                        for b in batch
                    )
                elif isinstance(batch, dict):
                    batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                            for k, v in batch.items()}

                mse = self.cr.compute(batch)      # tensor
                total_mse += float(mse.item())    # accumulate as float
                count += 1

        avg_mse = total_mse / max(1, count)
        self.model.train()
        return {"mse": avg_mse}
        
    def print_process(self, *x):
        if self.verbose:
            print(*x)
