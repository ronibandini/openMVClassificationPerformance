# OpenMV EI Image Classification Performance
# Roni Bandini July 2024 @ronibandini
# https://bandini.medium.com

import sensor, image, time, os, tf, uos, gc, machine, utime

# long delay to test idle power consumption

time.sleep_ms(6000)

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.SXGA)
sensor.set_windowing(1024,1024)
sensor.skip_frames(time=2000)          # Let the camera adjust

net = None
labels = None

print("OpenMV RT1062 Edge Impulse Image Classification Performance")
print("Roni Bandini - July 2024")
print("")

try:
    # load the model, alloc file on the heap if we have at least 64K free after loading
    net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    print(e)
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()
    start=utime.ticks_ms()

    for obj in net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        # Combines the labels and confidence values into a list of tuples
        predictions_list = list(zip(labels, obj.output()))

        for i in range(len(predictions_list)):
            print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))

    end =utime.ticks_ms()
    print(clock.fps(), "fps")
    print(end-start,"ms")




