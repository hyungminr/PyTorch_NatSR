import os
import torch
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(data.Dataset):
    def __init__(self, datadir, transforms, train=True):
        self.dataset = datadir
        self.transforms = transforms
        self.train = train
        if self.train:
            self.images = self.find_files('Train')
        else:
            self.images = self.find_files('Demo_Validation')
        
    def find_files(self, mode):
        filenames = []
        for (path, _, files) in os.walk(self.dataset+mode):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext.lower() in ['.jpg', '.jpeg', '.png']:
                    filenames.append(f'{path}/{filename}')
        return filenames
            
    def __getitem__(self, index):
        
        def load_tensor(filename, transform=self.transforms[0]):
            image = Image.open(filename)
            return transform(image)

        def indexerror(index, dataset):
            while index >= len(dataset):
                index -= len(dataset)
            return index
        
        index = indexerror(index, self.images)
        
        image_name = self.images[index]
        
        LR_tensor = load_tensor(image_name, self.transforms[0])                
        HR_tensor = load_tensor(image_name, self.transforms[1])
                
        return LR_tensor, HR_tensor
    
    def __len__(self):
        return len(self.images)

def get_loader(image_dir='./data/', batch_size=1, num_workers=1, train=True):
    
    def get_transform_hr(centercrop=448):        
        transform = []
        transform.append(T.CenterCrop(centercrop))
        transform.append(T.ToTensor())
        return T.Compose(transform)
    
    def get_transform_lr(centercrop=448):        
        transform = []
        transform.append(T.CenterCrop(centercrop))
        transform.append(T.Resize((centercrop//4, centercrop//4)))
        transform.append(T.ToTensor())
        return T.Compose(transform)

    transforms = []
    transforms.append(get_transform_lr())
    transforms.append(get_transform_hr())
    
    shuffle = (train == True)
    
    data_loader = torch.utils.data.DataLoader(dataset=Dataset(image_dir, transforms, train),
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    
    """
    ####### How to Use #######
    
    data_loader = get_loader(Train=True)
    data_iter = iter(data_loader)
    
    ...
    
    try:
        LR_tensor, HR_tensor = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        LR_tensor, HR_tensor = next(data_iter)
    
    """
    
    return data_loader