# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import cv2
import time

# Open live video stream on webcam at first index(i.e. 0) device
stream = VideoGear(source=0).start()

output_params = {"-input_framerate": 30,  "-vcodec": "libx265", "-video_size": "1280x720", "-preset":"fast", "-b:v":"4M", "-maxrate":"4M", "-minrate":"4M", "-bufsize":"4M"}

# Define writer with default parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output_filename=str(1)+".mp4", **output_params)

# loop over
start = time.time()
count=1
while True:
    
    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # simulating RGB frame for example
    frame_rgb = frame[:, :, ::-1]
    if (time.time() - start) > 60:
    # writing RGB frame to writer
        start = time.time()
        count+=1
        writer = WriteGear(output_filename=str(count)+".mp4", **output_params)
        writer.write(frame_rgb, rgb_mode=True)  # activate RGB Mode
        
    writer.write(frame_rgb, rgb_mode=True) 
    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
end = time.time()
# close output window
cv2.destroyAllWindows()
print("Time Difference : ", (end-start))
# safely close video stream
stream.stop()
# print(time.ctime())

# safely close writer
writer.close()