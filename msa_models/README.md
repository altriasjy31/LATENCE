# profun-som

### Description
predicting protein function from multiple sequence alignment

### Software Architecture

> profun-som-dev/<br>
> configs <br>
>> som # yaml format model architecture parameters

> experiments <br>
>> go # build_som.py and GO.py for building som model and MSA encoder <br>
>> preprocess # alignment.py, alignparser.py, sequence.py and utils.py

> helper_functions # helper.py <br>
> models # architecture.py gendis.py resnet.py utils.py <br>
> scripts # calculate_msa.py and predict.py <br>

#### Instructions
> python predict.py --help

> usage: predict.py [-h] [--msa-buffer-size MSA_BUFFER_SIZE] [--MAXLEN MAXLEN] [--top-k TOP_K] [--gpu-ids GPU_IDS] <br>
>                  fapath dbpath hhlib configdir weightpaths weightpaths weightpaths golabelpath saving_path
>> positional arguments:

>>>  fapath <br>
>>>  dbpath <br>
>>>  hhlib <br>
>>>  configdir <br>
>>>  weightpaths <br>
>>>> cco mfo bpo weight paths

>>>  golabelpath <br>
>>>> task_go_ordered_lst pkl file path

>>>  saving_path <br>
>>>> prediction result saving path

>> optional arguments:
>>>  -h, --help <br>
>>>> show this help message and exit

>>>  --msa-buffer-size
>>>> MSA_BUFFER_SIZE
>>>> the maximum buffer size to read sequence from MSA for sampling

>>>  --MAXLEN MAXLEN   
>>>> the maximum length for sequence

>>>  --top-k TOP_K
>>>> the maximum sampling size for input

>>>  --gpu-ids GPU_IDS
>>>> set the device, e.g. 0,1 or 0, where -1 means cpu

1.  Assign the absolute path of <b> the fasta format file of the query protein </b>, <b> the dataset for alignment </b>, <b> the hhblits install directory </b>, <b> model config directory </b>,  <b> model weight files (cco, mfo, and bpo)</b>, <b> a pickle format file of the go information </b> and <b> a target file for saving the predictions</b> to the parameter <b> fapath </b>, <b> dbpath </b>, <b> hhlib </b> <b> configdir </b>, <b> weightpaths </b>, <b> golabelpath </b>, and <b> saving_path </b>
2. The MSA_BUFFER_SIZE = 10000, MAXLEN = 2000, TOP_K = 40 in default
3. In the ubuntu 18.04 LTS, we could set these parameters by environment variables

<ul>
<li> python predict.py \${QUERY} \</li>
<li>\${DB} \</li>
<li>\${HHLIB} \</li>
<li>config/som \</li>
<li>\${CONFIG} \</li>
<li>\${CCO} \</li>
<li>\${MFO} \</li>
<li>\${BPO} \</li>
<li>\${GO} \</li>
<li>${SAVING}</li>
</ul>
