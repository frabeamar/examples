# examples
A collection of notes / tutorial and examples. Mostly deeplearning. 
Might not be pretty
## diffusers
trained ddpm via huggingface on cifar. 
## tensor_rt

JIT compiler -> just in time; only the parts that run often get optimize
In some update for pytorch they introduced torch script to make the models compilable, it uses jit (just in time compiler), it is similar to tensorflow graph mode.
This removes the python dependencies that the graph may have
there are two ways to generate a model with tensorRT 
 - tracing  : follow execution of the model, and generate tensor rt ops which replicate the behaviour
 - scripting : analyze the code; this can add control flow, which scripting cannot do.
 Tracing is easier

??? maybe true => ai
Torchscript is being phased out as it has several problems:
 - limited python support
 - criptic messages for debugging
 now you use torch.export to create an itermediate representation, a flattened computational graph with low level op. This can be then compiled to get a platform specific  binary

## kuber_docker
Orchestrate kubernetis locally via minikube
