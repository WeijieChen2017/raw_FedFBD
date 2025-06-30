import set_path
import warnings
import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import INFO
from flwr.common import Context
from experiments.MedMNIST2D.models import ResNet18, ResNet50
from torchvision.models import resnet18, resnet50
import torchvision.transforms as transforms
import PIL

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(data_flag, num_clients, batch_size, resize, as_rgb, size):
    """Load the MedMNIST dataset and partition it."""
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # Transformations
    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])

    # Load entire dataset
    train_dataset = DataClass(split='train', transform=data_transform, download=True, as_rgb=as_rgb, size=size)
    val_dataset = DataClass(split='val', transform=data_transform, download=True, as_rgb=as_rgb, size=size)
    
    # Partitioning
    train_len = len(train_dataset)
    val_len = len(val_dataset)
    train_indices = list(range(train_len))
    val_indices = list(range(val_len))
    
    train_partition_size = train_len // num_clients
    val_partition_size = val_len // num_clients

    train_partitions = []
    val_partitions = []
    for i in range(num_clients):
        train_start = i * train_partition_size
        train_end = (i + 1) * train_partition_size if i < num_clients - 1 else train_len
        val_start = i * val_partition_size
        val_end = (i + 1) * val_partition_size if i < num_clients - 1 else val_len
        
        train_partitions.append(Subset(train_dataset, train_indices[train_start:train_end]))
        val_partitions.append(Subset(val_dataset, val_indices[val_start:val_end]))
        
    trainloaders = [DataLoader(part, batch_size=batch_size, shuffle=True) for part in train_partitions]
    valloaders = [DataLoader(part, batch_size=batch_size) for part in val_partitions]
    
    test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=as_rgb, size=size)
    testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return trainloaders, valloaders, testloader

def get_model(model_flag, data_flag, resize, as_rgb):
    """Return a model for MedMNIST."""
    info = INFO[data_flag]
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    if model_flag == 'resnet18':
        model = resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model = resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError
    return model

def train(model, train_loader, epochs, task):
    """Train the model on the training set."""
    if task == "multi-label, binary-class":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = torch.squeeze(targets, 1).long()
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def test(model, data_loader, task, save_folder=None, run_name=None):
    """Validate the model on the test set."""
    if task == "multi-label, binary-class":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    model.eval()
    total_loss = 0
    y_score = torch.tensor([]).to(DEVICE)
    data_labels = torch.tensor([]).to(DEVICE)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
                m = torch.nn.Sigmoid()
                outputs = m(outputs)
            else:
                targets_squeezed = torch.squeeze(targets, 1).long()
                loss = criterion(outputs, targets_squeezed)
                m = torch.nn.Softmax(dim=1)
                outputs = m(outputs)

            total_loss += loss.item() * inputs.size(0)
            y_score = torch.cat((y_score, outputs), 0)
            data_labels = torch.cat((data_labels, targets), 0)

    y_score = y_score.detach().cpu().numpy()
    data_labels = data_labels.detach().cpu().numpy()
    
    evaluator = medmnist.Evaluator(data_loader.dataset.dataset.info['flag'], data_loader.dataset.dataset.split)
    auc, acc = evaluator.evaluate(y_score, save_folder=save_folder, run=run_name)
    
    loss = total_loss / len(data_loader.dataset)
    return loss, {"accuracy": acc, "auc": auc}

class FlowerClient(fl.client.NumPyClient):
    """Flower client for MedMNIST."""
    def __init__(self, model, trainloader, valloader, task):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.task = task

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1, task=self.task)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = test(self.model, self.valloader, self.task)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(metrics["accuracy"]), "auc": float(metrics["auc"])}

def client_fn(model_flag, data_flag, resize, as_rgb, trainloader, valloader):
    """Create a Flower client."""
    model = get_model(model_flag, data_flag, resize, as_rgb).to(DEVICE)
    task = INFO[data_flag]['task']
    return FlowerClient(model, trainloader, valloader, task).to_client() 