# Get Similarity Matrix
The following scripts are used to get similarity matrix from a given dataset. 

Different models can be used, defined in `models.py`.

### Models

#### *1. VGG architecure, trained on VGGFace2 dataset*

Pre-trained on 8749 identities from the VGGFace2 dataset.
When using this model, weights should be provided.

#### *2. VGG architecure, trained on imagenet dataset*
   
Model available online, no weights needed.

#### *3. InceptionResNet architecture* 

Use the one of the two options to get the pre-trained model:
vggface2, casia-webface
   
See documentation here: https://github.com/timesler/facenet-pytorch


### Data 

The data should be organized in the following way:

- main_folder
  - id1
      - img1
      - img2
      - ...
  - id2
      - img1
      - img2
      - ...


### Usage

To get the similarity matrix with the default model (resnet pretrained on vggface2), run the following command:
```bash
python similarity/rdm.py --data_dir <path_to_data> --output_path <path_to_results_dir>
```
To experiment with different models, add the arguments `--model_type` with one of the following options:
- vgg_vggface2 
- vgg_imagenet 
- resnet_vggface2 (default)
- resnet_casia

See Models section above for more details.

Note: if type chosen is vgg_vggface2, `--model_path` should be provided with local path to weights.

```bash
python similarity/rdm.py --data_dir <path_to_data> --output_path <path_to_results_dir> --model_type vgg_vggface2 --model_path <path_to_weights>
```
