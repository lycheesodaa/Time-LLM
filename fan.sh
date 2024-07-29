# sources:
# https://forums.developer.nvidia.com/t/nvidia-settings-returns-connection-refused-error-the-control-display-is-undefined/173530/4
# https://askubuntu.com/questions/42494/how-can-i-change-the-nvidia-gpu-fan-speed

if [ $# -eq 0 ]
  then
    echo "Please enter a fan speed, e.g. for 60% enter 60"
fi

echo "Start a new X server"
X :0 &

export DISPLAY=:0

sleep 15

#### start your settings here. 
#### You can use both 'nvidia-smi' and 'nvidia-settings'

#fan speed
echo .
echo "Set fans"

nvidia-settings -a [gpu:0]/GPUFanControlState=1 -a [fan:0]/GPUTargetFanSpeed=$1 -a [fan:1]/GPUTargetFanSpeed=$1
nvidia-settings -a [gpu:1]/GPUFanControlState=1 -a [fan:2]/GPUTargetFanSpeed=$1 -a [fan:3]/GPUTargetFanSpeed=$1
