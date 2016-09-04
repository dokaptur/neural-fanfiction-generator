# neural-fanfiction-generator
Encoder-decoder RNN model for automatic fanfiction generation. Created by Dorota Kołodziejska ([dokaptur](https://github.com/dokaptur)) and Guillaume Bouchard ([gbouchar](https://github.com/gbouchar))

## Prerequisits
1. Python 3.5
2. tensorflow 0.9
3. nltk 3.0

## Usage
Before using any code from this repository it's important to set the environmental variable PYTHONPATH to the root of this project. 
### Generating data
To train this model one needs data, that is corpora of fanfictions or other stories. If you don't have your own data, you can generate simulated fanfictions using python script *scripts/generate_simulations.py*. It takes two positional arguments:
output directory for generated simulations and the number of stories to generate. 

Example:
```bash
python scripts/generate_simulations.py /tmp/data 500
```
It will create an output directory if it does not already exist and create required number of stories, each in one file.

### Preprocessing and data preparation
The model needs tokenized stories (both training and validation set), vocabulary and the information for each word if it's a named entity and thus should be generated using uppercase. Additionally, the directory containing data should be structured as follows:
```
|-- vocabulary
|-- vocabulary_uppercase
|-- stories
    |-- <story1>
    |-- <story2>
    ...
|-- test
    |-- stories
        |-- <story1>
        |-- <story2>
        ...
```
The script *scripts/prepare_data.py* prepares the data in a way described above. It takes
four positional arguments: a directory containing stories (like tho one created in a previous step), output directory for processed data, size of the vocabulary to create (it takes *n* most popular words, the rest is replaced later by the UNKNOWN token) and the number of stories for validation set.

Example:
```
python scripts/prepare_data.py /tmp/data /tmp/simulations 1000 50
```

### Training
To train the model one can use the script *scripts/learn.py*. It takes two positional arguments: directory containing data (structured as by the *prepare_data* script) and output directory for the trained model. Additionally it takes a lot of optional arguments which can change values of most of the model parameters. Type
```
python scripts/learn.py --help
```
to learn more about all of optional parameters.

Example:
```
python scripts/learn.py /tmp/simualtions /tmp/model --warm_start --n_layers=2 --hidden_size=31 --emb_dim=11
```

### Generating stories
Once the model has been trained, one can generate a brand new story using the script *scripts/generate.py*. It takes the same positional arguments as the learning script. Data directory is needed for vocabulary files and a path to the trained model should contain files created by tensorflow while saving the graph. It's important that the files follow the naming convention used in this repo, that is model name should contain information about model parameters. 

Additionally one can use optional parameters to determine sampling and set the first sentence of the generated story.

Example:
```
python scripts/generate.py /tmp/simualtions /tmp/model
```

