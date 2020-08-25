import tensorflow as tf
import syft_tensorflow as sy

hook = sy.TensorFlowHook(tf)
# Simulates a remote worker (ie another computer)
remote = sy.VirtualWorker(hook, id="remote")

# Send data to the other worker
x = tf.constant(5).send(remote)
y = tf.constant(10).send(remote)

z = x * y

print(z.get())
# => 50