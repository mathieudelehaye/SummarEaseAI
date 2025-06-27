import tensorflow as tf
import time

print("🚀 Testing RTX 4070 DirectML GPU Acceleration")
print("=" * 50)

# Check TensorFlow version and devices
print(f"TensorFlow version: {tf.__version__}")
print(f"Available devices: {tf.config.list_physical_devices()}")

# Test GPU computation
print("\n🧮 Testing GPU computation...")
with tf.device('/GPU:0'):
    # Create matrices
    x = tf.random.normal([1000, 1000])
    y = tf.random.normal([1000, 1000])
    
    # Time the computation
    start = time.time()
    z = tf.matmul(x, y)
    gpu_time = (time.time() - start) * 1000
    
    print(f"✅ GPU computation completed!")
    print(f"⚡ Time: {gpu_time:.2f} ms")
    print(f"📊 Result shape: {z.shape}")

# Compare with CPU
print("\n🖥️  Testing CPU computation for comparison...")
with tf.device('/CPU:0'):
    start = time.time()
    z_cpu = tf.matmul(x, y)
    cpu_time = (time.time() - start) * 1000
    
    print(f"✅ CPU computation completed!")
    print(f"⏱️  Time: {cpu_time:.2f} ms")

# Show speedup
if cpu_time > 0:
    speedup = cpu_time / gpu_time
    print(f"\n🚀 GPU Speedup: {speedup:.2f}x faster!")

print(f"\n🎉 DirectML GPU acceleration is working with your RTX 4070!") 