


::: {.cell .markdown}

## Launch and set up NVIDIA A100 40GB server - with python-chi

At the beginning of the lease time, we will bring up our GPU server. We will use the `python-chi` Python API to Chameleon to provision our server. 

> **Note**: if you don't have access to the Chameleon Jupyter environment, or if you prefer to set up your NVIDIA server by hand, the next section provides alternative instructions! If you want to set up your server "by hand", skip to the next section.


We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected:

:::

::: {.cell .code}
```python
from chi import server, context, lease
import os

context.version = "1.0" 
context.choose_project()
context.choose_site(default="CHI@TACC")
```
:::

::: {.cell .markdown}

Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:

:::

::: {.cell .code}
```python
l = lease.get_lease(f"mltrain_netID") 
l.show()
```
:::

::: {.cell .markdown}

The status should show as "ACTIVE" now that we are past the lease start time.

The rest of this notebook can be executed without any interactions from you, so at this point, you can save time by clicking on this cell, then selecting "Run" > "Run Selected Cell and All Below" from the Jupyter menu.  

As the notebook executes, monitor its progress to make sure it does not get stuck on any execution error, and also to see what it is doing!

:::

::: {.cell .markdown}

We will use the lease to bring up a server with the `CC-Ubuntu24.04-CUDA` disk image. 

> **Note**: the following cell brings up a server only if you don't already have one with the same name! (Regardless of its error state.) If you have a server in ERROR state already, delete it first in the Horizon GUI before you run this cell.

:::


::: {.cell .code}
```python
username = os.getenv('USER') # all exp resources will have this prefix
s = server.Server(
    f"node-mltrain-{username}", 
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-CUDA"
)
s.submit(idempotent=True)
```
:::

::: {.cell .markdown}

Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.

:::

::: {.cell .markdown}

Then, we'll associate a floating IP with the instance, so that we can access it over SSH.

:::

::: {.cell .code}
```python
s.associate_floating_ip()
```
:::

::: {.cell .code}
```python
s.refresh()
s.check_connectivity()
```
:::

::: {.cell .markdown}

In the output below, make a note of the floating IP that has been assigned to your instance (in the "Addresses" row).

:::

::: {.cell .code}
```python
s.refresh()
s.show(type="widget")
```
:::




::: {.cell .markdown}

## Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance, to set it up. We'll start by retrieving the code and other materials on the instance.

:::

::: {.cell .code}
```python
s.execute("git clone --recurse-submodules https://github.com/teaching-on-testbeds/MLOps-Final-Project")
```
:::


::: {.cell .markdown}

## Set up Docker

To use common deep learning frameworks like Tensorflow or PyTorch, and ML training platforms like MLFlow and Ray, we can run containers that have all the prerequisite libraries necessary for these frameworks. Here, we will set up the container framework.

:::

::: {.cell .code}
```python
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
```
:::

::: {.cell .markdown}

## Set up the NVIDIA container toolkit


We will also install the NVIDIA container toolkit, with which we can access GPUs from inside our containers.

:::

::: {.cell .code}
```python
s.execute("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list")
s.execute("sudo apt update")
s.execute("sudo apt-get install -y nvidia-container-toolkit")
s.execute("sudo nvidia-ctk runtime configure --runtime=docker")
# for https://github.com/NVIDIA/nvidia-container-toolkit/issues/48
s.execute("sudo jq 'if has(\"exec-opts\") then . else . + {\"exec-opts\": [\"native.cgroupdriver=cgroupfs\"]} end' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp > /dev/null && sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json")
s.execute("sudo systemctl restart docker")
```
:::

::: {.cell .markdown}

and we can install `nvtop` to monitor GPU usage:

:::


::: {.cell .code}
```python
s.execute("sudo apt update")
s.execute("sudo apt -y install nvtop")
```
:::





::: {.cell .markdown}

###  Build a container image - for MLFlow section


Finally, we will build a container image in which to work in the MLFlow section, that has:

* a Jupyter notebook server
* Pytorch and Pytorch Lightning
* CUDA, which allows deep learning frameworks like Pytorch to use the NVIDIA GPU accelerator
* and MLFlow

You can see our Dockerfile for this image at: [Dockerfile.jupyter-torch-mlflow-cuda](https://github.com/teaching-on-testbeds/MLOps-Final-Project/training/tree/main/docker/Dockerfile.jupyter-torch-mlflow-cuda)


Building this container may take a bit of time, but that's OK: we can get it started and then continue to the next section while it builds in the background, since we don't need this container immediately.

:::


::: {.cell .code}
```python
s.execute("docker build -t jupyter-mlflow -f MLOps-Final-Project/training/docker/Dockerfile.jupyter-torch-mlflow-cuda .")
```
:::


::: {.cell .markdown}

Leave that cell running, and in the meantime, open an SSH sesson on your server. From your local terminal, run

```
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

where

* in place of `~/.ssh/id_rsa_chameleon`, substitute the path to your own key that you had uploaded to CHI@TACC
* in place of `A.B.C.D`, use the floating IP address you just associated to your instance.


:::
