import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
from config import (
    CMD, MODEL_NAME, WEIGHT_PATH_CLIENT, PORT_MAIN, FIXED_SPLIT_POINT,
    LR_ALPHA, L1_LAMBDA
)
from network_utils import ClientConnection, send_message, recv_message
from split_trainer import (
    SplitModelClient, extract_hyper_weights, load_hyper_weights, enforce_pruning
)

class ClientWorker:
    def __init__(self, server_ip, port, device='cuda'):
        self.server_ip = server_ip
        self.port = port
        self.device = device
        
        self.conn = None
        self.model = None
        self.split_model = None
        self.split_point = FIXED_SPLIT_POINT
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.pruned_weights = []
        self.client_hyper_params = []
    
    def connect(self):
        self.conn = ClientConnection(self.server_ip, self.port)
        self.conn.connect()
    def _create_model(self):
        from hyper_repvgg import create_hyper_repvgg
        print(f"[Client] Loading model with pretrained weights...")
        print(f"[Client] Weight path: {WEIGHT_PATH_CLIENT}")
        
        self.model = create_hyper_repvgg(WEIGHT_PATH_CLIENT, model_name=MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model.train()
        
        print(f"[Client] Model loaded: {MODEL_NAME}")
    
    def _setup_split_model(self, split_point):
        self.split_point = split_point
        self.split_model = SplitModelClient(self.model, split_point)
        self.client_hyper_params = self.split_model.get_hyper_params()

        for name, param in self.model.named_parameters():
            if 'hyper_' not in name:
                param.requires_grad = False
        
        print(f"[Client] Split point set to {split_point}, managing {len(self.client_hyper_params)} hyper params")
    
    def _handle_lut_measure(self, data):
        from lut_manager import LUTMeasurer
        print("\n[Client] LUT Measurement requested")
        
        if self.model is None:
            self._create_model()
        
        measurer = LUTMeasurer(self.model, device=self.device)
        results = measurer.measure_all_blocks()
        
        results_formatted = {}
        for block_name, gamma_dict in results.items():
            results_formatted[block_name] = {
                str(k): v for k, v in gamma_dict.items()
            }
        
        response = {
            'T_client': results_formatted,
            'metadata': {
                'device': 'client',
                'device_name': torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU',
                'build_time': str(datetime.now())
            }
        }
        
        send_message(self.conn.socket, CMD.LUT_RESULT, response)
        print("[Client] LUT measurement complete")
    
    def _handle_train_epoch_start(self, data):
        epoch = data['epoch']
        total_epochs = data['total_epochs']
        num_batches = data['num_batches']
        self.pruned_weights = data['pruned_weights']
        print(f"\n[Client] Training Epoch {epoch+1}/{total_epochs} starting ({num_batches} batches)")
        
        if self.model is None:
            self._create_model()
        
        self.model.train()
        
        send_message(self.conn.socket, CMD.ACK, None)
    
    def _handle_sync_model(self, data):
        hyper_state = data['hyper_state']
        split_point = data['split_point']

        load_hyper_weights(self.model, hyper_state, self.device)
        
        if self.split_model is None or self.split_point != split_point:
            self._setup_split_model(split_point)
        
        self.optimizer = optim.SGD(self.client_hyper_params, lr=LR_ALPHA, momentum=0.9)
        
        send_message(self.conn.socket, CMD.ACK, None)
    
    def _handle_forward_data(self, data):
        features = data['features'].to(self.device)
        labels = data['labels'].to(self.device)

        features.requires_grad = True
        
        self.optimizer.zero_grad()
        outputs = self.split_model.forward(features)
        
        loss = self.criterion(outputs, labels)
        
        l1_penalty = sum(torch.abs(p).sum() for p in self.client_hyper_params if p.requires_grad)
        total_loss = loss + L1_LAMBDA * l1_penalty
        
        total_loss.backward()
        
        grad = features.grad.cpu()
        
        self.optimizer.step()
        
        enforce_pruning(self.model, self.pruned_weights)
        
        send_message(self.conn.socket, CMD.BACKWARD_GRAD, {
            'grad': grad,
            'loss': loss.item()
        })
    
    def _handle_train_epoch_end(self, data):
        epoch = data['epoch']

        client_stages = self.split_model.get_client_stages()
        
        client_hyper_state = OrderedDict()
        for name, param in self.model.named_parameters():
            if 'hyper_' in name:
                parts = name.split('.')
                stage = parts[0]
                block_idx = int(parts[1])
                
                is_after_split = False
                split_stage = self.split_model.split_stage
                split_block_idx = self.split_model.split_block_idx
                
                if stage == split_stage and block_idx > split_block_idx:
                    is_after_split = True
                elif stage == 'stage2' and split_stage == 'stage1':
                    is_after_split = True
                elif stage == 'stage3' and split_stage in ['stage1', 'stage2']:
                    is_after_split = True
                elif stage == 'stage4' and split_stage in ['stage1', 'stage2', 'stage3']:
                    is_after_split = True
                
                if is_after_split:
                    client_hyper_state[name] = param.data.cpu().clone()
        
        send_message(self.conn.socket, CMD.HYPER_WEIGHTS, {
            'hyper_state': client_hyper_state
        })
        
        print(f"[Client] Epoch {epoch+1} complete, sent {len(client_hyper_state)} hyper weights to Server")
    
    def _handle_domain_start(self, data):
        domain_id = data['domain_id']
        print(f"\n{'='*60}")
        print(f"[Client] Domain {domain_id} started")
        print(f"{'='*60}")
    def _handle_domain_end(self, data):
        domain_id = data['domain_id']
        print(f"[Client] Domain {domain_id} completed")
    def _handle_evaluate_start(self, data):
        num_batches = data['num_batches']
        print(f"\n[Client] Starting evaluation ({num_batches} batches)...")

        if self.model is None:
            self._create_model()
        
        if self.split_model is None:
            self._setup_split_model(self.split_point)
        
        self.model.eval()
        
        send_message(self.conn.socket, CMD.ACK, {})
    
    def _handle_evaluate_batch(self, data):
        features = data['features'].to(self.device)
        labels = data['labels']
        with torch.no_grad():

            outputs = self.split_model.forward(features)
            _, predicted = torch.max(outputs.data, 1)
            
            predicted_cpu = predicted.cpu()
            correct = (predicted_cpu == labels).sum().item()
        
        send_message(self.conn.socket, CMD.EVALUATE_BATCH_RESULT, {
            'predictions': predicted_cpu.numpy().tolist(),
            'correct': correct,
            'batch_size': labels.size(0)
        })
    
    def _handle_evaluate_end(self, data):
        print(f"[Client] Evaluation completed")

        self.model.train()
        
        send_message(self.conn.socket, CMD.ACK, {})
    
    def _handle_comm_benchmark(self, data):
        send_message(self.conn.socket, CMD.FORWARD_DATA, torch.tensor([1.0]))
    def run(self):
        print("\n" + "="*60)
        print("Client Worker Started")
        print("="*60)
        print(f"Device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("="*60)

        self.connect()
        
        print("\n[Client] Listening for commands...")
        
        while True:
            try:
                cmd, data = recv_message(self.conn.socket)
                
                if cmd == CMD.SHUTDOWN:
                    print("\n[Client] Shutdown command received")
                    break
                
                elif cmd == CMD.LUT_MEASURE:
                    self._handle_lut_measure(data)
                
                elif cmd == CMD.TRAIN_EPOCH_START:
                    self._handle_train_epoch_start(data)
                
                elif cmd == CMD.SYNC_MODEL:
                    self._handle_sync_model(data)
                
                elif cmd == CMD.FORWARD_DATA:
                    if isinstance(data, dict) and 'features' in data:
                        self._handle_forward_data(data)
                    else:
                        self._handle_comm_benchmark(data)
                
                elif cmd == CMD.TRAIN_EPOCH_END:
                    self._handle_train_epoch_end(data)
                
                elif cmd == CMD.DOMAIN_START:
                    self._handle_domain_start(data)
                
                elif cmd == CMD.DOMAIN_END:
                    self._handle_domain_end(data)
                
                elif cmd == CMD.EVALUATE_START:
                    self._handle_evaluate_start(data)
                
                elif cmd == CMD.EVALUATE_BATCH:
                    self._handle_evaluate_batch(data)
                
                elif cmd == CMD.EVALUATE_END:
                    self._handle_evaluate_end(data)
                
                else:
                    print(f"[Client] Unknown command: {cmd}")
            
            except ConnectionError as e:
                print(f"\n[Client] Connection error: {e}")
                break
            
            except Exception as e:
                print(f"\n[Client] Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.conn.close()
        print("\n[Client] Worker stopped")

def main():
    parser = argparse.ArgumentParser(description='Joint Optimization - Client Worker')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=PORT_MAIN)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        args.device = 'cpu'
    
    worker = ClientWorker(args.server_ip, args.port, args.device)
    worker.run()

if __name__ == '__main__':
    main()