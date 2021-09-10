# muss abgeändert werden für andere Systeme
personal_path = "C:/Users/praktikant/"


# youtubevideo to installation  = https://www.youtube.com/watch?v=0-4p_QgrdbE

# bei belieben Abändern
Ordnername = "ANPR" # muss der Ordnername sein, wo diese Datei liegt
Venvname = "anprsys"


import os
# pathsSetup
# global variables
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

default_path = f"{Ordnername}/TFODCourse"
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
}
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}
labels = [{'name':'licence', 'id':1}]
with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

def installPackage(Package):
    try:
        os.system(f"pip install {Package} --no-cache-dir")
    except:
        print(f"Could not install {Package}")



# Step 1
def step1():
    earlySetup()
    createPaths()
    downloadTFModels()
    verificationScript()#try to run it more than once, it can happen that some dependencies are missing
    install_after_verification()
    import object_detection
    pretrained_model_copy()

########################################################################################################################

def earlySetup():
    os.system(f"move {personal_path+default_path}\* {personal_path+Ordnername}")
    os.system(f"move {personal_path+default_path}\.* {personal_path+Ordnername}")
    os.system(f"move {personal_path+default_path}\.ipynb_checkpoints {personal_path+Ordnername}")
    os.system(f"rmdir /S /Q TFODCourse")
    
    print("create Venv")
    os.system(f"cd {personal_path+default_path}")
    os.system("py -3.8 -m venv {Venvname}")
    os.system(f"./{Venvname}/Scripts/activate") # startet das Virtual Environment
    os.system("py -3.8 -m pip install --upgrade pip")
    os.system("pip install ipykernel")
    os.system(f"py -3.8 -m ipykernel install --user --name={Venvname}")


def createPaths():
    print("Setting up Folderstructure")
    for path in paths.values():
        if not os.path.exists(path):
            # LINUX
            if os.name == 'posix':
                os.system(f"mkdir -p {path}")
            # WINDOWS
            if os.name == 'nt':
                os.system(f"mkdir {path}")


def downloadTFModels():
    installPackage("wheel")
    installPackage("wget")
    installPackage("setuptools")

    print("Clone Pretrained Model")
    if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
        os.system(f"git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}")

    # same as "pip list" but can be used instead of a console output
    from pip._internal.utils.misc import get_installed_distributions
    installed_packages = [str(i).split(" ")[0] for i in get_installed_distributions()]
    
    needed_packages = ['google-resumable-media', 'google-cloud-core', 'google-cloud-bigquery', 'google-auth-httplib2', 'future', 'dm-tree', 'dill', 'dataclasses', 'clang', 'zipp', 'wrapt', 'widgetsnbextension', 'Werkzeug', 'webencodings', 'wcwidth', 'urllib3', 'uritemplate', 'typeguard', 'twine', 'traitlets', 'tqdm', 'tornado', 'tomli', 'tifffile', 'threadpoolctl', 'text-unidecode', 'testpath', 'terminado', 'termcolor', 'smart-open', 'six', 'setuptools', 'Send2Trash', 'scikit-learn', 'scikit-image', 'rsa', 'rfc3986', 'requests-toolbelt', 'requests-oauthlib', 'readme-renderer', 'QtPy', 'qtconsole', 'pyzmq', 'pywinpty', 'pywin32', 'pywin32-ctypes', 'PyWavelets', 'python-slugify', 'python-dateutil', 'python-bidi', 'pyrsistent', 'PyQt5', 'PyQt5-sip', 'PyQt5-Qt5', 'pyparsing', 'Pygments', 'pycparser', 'pycocotools', 'pyasn1', 'pyasn1-modules', 'protobuf', 'proto-plus', 'prompt-toolkit', 'promise', 'prometheus-client', 'pkginfo', 'pickleshare', 'parso', 'pandocfilters', 'opt-einsum', 'oauthlib', 'notebook', 'networkx', 'nest-asyncio', 'nbformat', 'nbconvert', 'nbclient', 'mistune', 'matplotlib', 'matplotlib-inline', 'MarkupSafe', 'Markdown', 'keyring', 'keras', 'Keras-Preprocessing', 'jupyterlab-widgets', 'jupyterlab-pygments', 'jupyter', 'jupyter-core', 'jupyter-console', 'jupyter-client', 'jsonschema', 'joblib', 'Jinja2', 'jedi', 'ipywidgets', 'ipython', 'ipython-genutils', 'importlib-resources', 'importlib-metadata', 'imageio', 'idna', 'httplib2', 'h5py', 'grpcio', 'googleapis-common-protos', 'google-pasta', 'google-crc32c', 'google-auth', 'google-auth-oauthlib', 'google-api-core', 'gensim', 'gast', 'flatbuffers', 'entrypoints', 'docutils', 'defusedxml', 'decorator', 'debugpy', 'Cython', 'colorama', 'charset-normalizer', 'cffi', 'certifi', 'cachetools', 'bleach', 'backcall', 'attrs', 'astunparse', 'argon2-cffi', 'absl-py', 'object-detection', 'pandas', 'scipy', 'lvis', 'contextlib2', 'lxml', 'pillow', 'apache-beam', 'avro-python3', 'tensorflow-text', 'seqeval', 'sentencepiece', 'sacrebleu', 'pyyaml', 'py-cpuinfo', 'psutil', 'opencv-python-headless', 'oauth2client', 'kaggle', 'google-api-python-client', 'gin-config', 'pytz', 'opencv-python', 'kiwisolver', 'cycler', 'setuptools-scm', 'packaging', 'fonttools', 'typing-extensions', 'requests', 'pymongo', 'pydot', 'pyarrow', 'orjson', 'slim']
    missing_packages = [i for i in needed_packages if i not in installed_packages]
    print(f"installing {len(missing_packages)} of {len(needed_packages)} needed Packages")
    for package in missing_packages:
        print(f"Installing Package: {package}")
        installPackage(package)

    # Install Tensorflow Object Detection
    print("Install Tensorflow")
    # LINUX
    if os.name=='posix':  
        os.system("apt-get install protobuf-compiler")
        os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .") 
    
    # WINDOWS
    if os.name=='nt':
        import wget
        url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
        wget.download(url)
        os.system(f"move protoc-3.15.6-win64.zip {paths['PROTOC_PATH']}")
        os.system(f"cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip")
        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))
        os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install")
        os.system("cd Tensorflow/models/research/slim && pip install -e .")


def install_after_verification():
    to_install = ["Pillow","pytz"]
    for package in to_install:
        installPackage(package)
    os.system("pip uninstall --yes numpy")
    os.system("pip install numpy==1.19.5")
    os.system("pip uninstall --yes tf-models-official")
    os.system("pip install ./tf_models_official-2.4.0-py2.py3-none-any.whl") # will install latest Version of Tensorflow which we dont want
    os.system("pip uninstall --yes tensorflow")
    os.system("pip install ./tensorflow-2.4.1-cp38-cp38-win_amd64.whl")
    
    os.system("pip uninstall matplotlib -y")
    os.system("pip install matplotlib==3.2.2 --no-cache-dir")


def verificationScript():
    VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    # Verify Installation
    os.system(f"python {VERIFICATION_SCRIPT}")
    print("Verification Script ran")


def pretrained_model_copy():
    print("Copy Pretrained Model")
    import wget
    if os.name =='posix':
        os.system(f"wget {PRETRAINED_MODEL_URL}")
        os.system(f"mv {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}")
        os.system(f"cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}")

    if os.name == 'nt':
        wget.download(PRETRAINED_MODEL_URL)
        os.system(f"move {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}")
        os.system(f"cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}")

########################################################################################################################
######################                         End of Step 1                               #############################
########################################################################################################################



# Step 2
def step2():
    createLabelMap()
    copyImages()

########################################################################################################################


def createLabelMap():
    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

def copyImages():
    print("Copying Images")
    os.system(f"move /y test {os.path.join(paths['IMAGE_PATH'])}")
    os.system(f"move /y train {os.path.join(paths['IMAGE_PATH'])}")

########################################################################################################################
######################                         End of Step 2                               #############################
########################################################################################################################



# Step 3
def step3():
    createTFRecords()
    copyTFRecords()

########################################################################################################################

def createTFRecords():
    ## OPTIONAL IF RUNNING ON COLAB
    #ARCHIVE_FILES = os.path.join(paths['IMAGE_PATH'], 'archive.tar.gz')
    #if os.path.exists(ARCHIVE_FILES):
    #    os.system(f"tar -zxvf {ARCHIVE_FILES}")
    ################################

    print("Cloning TFRecord")
    if not os.path.exists(files['TF_RECORD_SCRIPT']):
        os.system(f"git clone https://github.com/nicknochnack/GenerateTFRecord {paths['SCRIPTS_PATH']}")

    "Running TFRecord Script"
    os.system(f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}")
    os.system(f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}")

def copyTFRecords():
    print("Copying TFRecords")
    os.system(f"move /y test.record {os.path.join(paths['ANNOTATION_PATH'])}")
    os.system(f"move /y train.record {os.path.join(paths['ANNOTATION_PATH'])}")


########################################################################################################################
######################                         End of Step 3                               #############################
########################################################################################################################



# Step 4
def step4():

    if os.name =='posix':
        os.system(f"cp {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}")
    if os.name == 'nt':
        os.system(f"copy {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}")


########################################################################################################################
######################                         End of Step 4                               #############################
########################################################################################################################




# Step 5
def step5():

    import tensorflow as tf
    from object_detection.utils import config_util
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format
    config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)  

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text)   


########################################################################################################################
######################                         End of Step 5                               #############################
########################################################################################################################



step1()
step2()
step3()
step4()
step5()