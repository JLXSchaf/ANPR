import os
# youtubevideo to installation  = https://www.youtube.com/watch?v=0-4p_QgrdbE
# create Annotations for own images = https://www.makesense.ai/


# Ordner erstellen und Datei hineinziehen und ausführen
# bei belieben Abändern
Ordnername = "ANPRScript"
Venvname = "anprscript"

# muss abgeändert werden für andere Systeme
personal_path = "C:/Users/praktikant/"


# pathsSetup
# needed all the time
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
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


def installPackage(Package):
    try:
        os.system(f"pip install {Package}")
    except:
        print(f"Could not install {Package}")


def earlySetup():
    default_path = f"{Ordnername}/TFODCourse"
    os.system(f"move {personal_path+default_path}\* {personal_path+Ordnername}")
    os.system(f"move {personal_path+default_path}\.* {personal_path+Ordnername}")
    os.system(f"move {personal_path+default_path}\.ipynb_checkpoints {personal_path+Ordnername}")
    os.system(f"rmdir /S /Q TFODCourse")
    
    print("create Venv")
    os.system("python -m venv {Venvname}")
    os.system(f"./{Venvname}/Scripts/activate") # startet das Script
    os.system("python -m pip install --upgrade pip")
    os.system("pip install ipykernel")
    os.system(f"python -m ipykernel install --user --name={Venvname}")


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
    os.system("pip install wget")
    import wget

    print("Clone Pretrained Model")
    if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
        os.system(f"git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}")

    # same as "pip list" but can be used instead of a console output
    from pip._internal.utils.misc import get_installed_distributions
    installed_packages = [str(i).split(" ")[0] for i in get_installed_distributions()]
    
    needed_packages = ['pandas', 'absl-py', 'anyio', 'apache-beam', 'argon2-cffi', 'astunparse', 'attrs', 'avro-python3', 'Babel', 'backcall', 'bleach', 'cachetools', 'certifi', 'cffi', 'charset-normalizer', 'clang', 'colorama', 'contextlib2', 'crcmod', 'cycler', 'Cython', 'debugpy', 'decorator', 'defusedxml', 'dill', 'dm-tree', 'docopt', 'docutils', 'easyocr', 'entrypoints', 'fastavro', 'flatbuffers', 'future', 'gast', 'gin', 'gin-config', 'google-api-core', 'google-api-python-client', 'google-auth', 'google-auth-httplib2', 'google-auth-oauthlib', 'google-pasta', 'googleapis-common-protos', 'grpcio', 'h5py', 'hdfs', 'httplib2', 'idna', 'imageio', 'importlib-metadata', 'ipykernel', 'ipython', 'ipython-genutils', 'ipywidgets', 'jedi', 'Jinja2', 'joblib', 'json5', 'jsonschema', 'jupyter', 'jupyter-client', 'jupyter-console', 'jupyter-core', 'jupyter-server', 'jupyterlab', 'jupyterlab-pygments', 'jupyterlab-server', 'jupyterlab-widgets', 'kaggle', 'keras', 'Keras-Preprocessing', 'keyring', 'kiwisolver', 'lvis', 'lxml', 'Markdown', 'MarkupSafe', 'matplotlib', 'matplotlib-inline', 'mistune', 'nbclassic', 'nbclient', 'nbconvert', 'nbformat', 'nest-asyncio', 'networkx', 'notebook', 'numpy', 'oauth2client', 'oauthlib', 'object-detection', 'opencv-python', 'opencv-python-headless', 'opt-einsum', 'orjson', 'packaging', 'pandas', 'pandocfilters', 'parso', 'pdiff', 'pickleshare', 'Pillow', 'pip', 'pkginfo', 'portalocker', 'prometheus-client', 'promise', 'prompt-toolkit', 'protobuf', 'psutil', 'py-cpuinfo', 'pyarrow', 'pyasn1', 'pyasn1-modules', 'pycocotools', 'pycparser', 'pydot', 'Pygments', 'pymongo', 'pyparsing', 'pyrsistent', 'python-bidi', 'python-dateutil', 'pytz', 'PyWavelets', 'pywin32', 'pywin32-ctypes', 'pywinpty', 'PyYAML', 'pyzmq', 'qtconsole', 'QtPy', 'readme-renderer', 'regex', 'requests', 'requests-oauthlib', 'requests-toolbelt', 'requests-unixsocket', 'rfc3986', 'rsa', 'sacrebleu', 'scikit-image', 'scikit-learn', 'scipy', 'Send2Trash', 'sentencepiece', 'seqeval', 'setuptools', 'six', 'slim', 'sniffio', 'tabulate', 'tensorboard', 'tensorboard-data-server', 'tensorboard-plugin-wit', 'tensorflow', 'tensorflow-addons', 'tensorflow-datasets', 'tensorflow-estimator', 'tensorflow-hub', 'tensorflow-metadata', 'tensorflow-model-optimization', 'tensorflow-object-detection-api', 'tensorflow-text', 'termcolor', 'terminado', 'testpath', 'text-unidecode', 'tf-models-official', 'tf-slim', 'threadpoolctl', 'tifffile', 'torch', 'torchaudio', 'torchvision', 'tornado', 'tqdm', 'traitlets', 'twine', 'typeguard', 'typing-extensions', 'uritemplate', 'urllib3', 'wcwidth', 'webencodings', 'websocket-client', 'Werkzeug', 'wget', 'wheel', 'widgetsnbextension', 'wrapt', 'zipp']
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
        url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
        wget.download(url)
        os.system(f"move protoc-3.15.6-win64.zip {paths['PROTOC_PATH']}")
        os.system(f"cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip")
        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))
        os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install")
        os.system("cd Tensorflow/models/research/slim && pip install -e .")


def install_after_verification():
    os.system("pip uninstall numpy")
    os.system("pip install numpy==1.19.5")
    os.system("pip install ./tf_models_official-2.4.0-py2.py3-none-any.whl") # will install latest Version of Tensorflow which we dont want
    os.system("pip uninstall tensorflow")
    os.system("pip install ./tensorflow-2.4.1-cp38-cp38-win_amd64.whl")
    to_install = ["Pillow", "", "wheel", "setuptools","pytz"]
    for package in to_install:
        installPackage(package)
    os.system("pip uninstall matplotlib -y")
    os.system("pip install matplotlib==3.2.2 --no-cache-dir")


def verificationScript():
    VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    # Verify Installation
    os.system(f"python {VERIFICATION_SCRIPT}")
    print("Verification Scripted ran")


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


def createLabelMap():
    labels = [{'name':'licence', 'id':1}]

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

def createTFrecords():
    # OPTIONAL IF RUNNING ON COLAB
    ARCHIVE_FILES = os.path.join(paths['IMAGE_PATH'], 'archive.tar.gz')
    if os.path.exists(ARCHIVE_FILES):
        os.system(f"tar -zxvf {ARCHIVE_FILES}")
    #############################

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

# Step 1
earlySetup()
createPaths()
downloadTFModels()
verificationScript()#try to run it more than once, it can happen that some dependencies are missing
install_after_verification()
import object_detection
pretrained_model_copy()

# Step 2
createLabelMap()
copyImages()

# Step 3
createTFrecords()
copyTFRecords()

# Step 4
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])