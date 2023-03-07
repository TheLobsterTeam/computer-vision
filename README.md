# Computer Vision
Computer vision scripts for aligning the wire to the via holes and calibrating overall PCB position.

## API usage

Initialize camera, returns None if there is an error starting the capture.

``` video = init_via() ```

Detect the wire and closest hole, returns the distance between them in mm, None if not found, or (0,0) if aligned.

``` adjustments = via_detection(video) ```

Cleanup opencv and release camera.

``` cleanup(video) ```

## Running the Program for Debugging

There are two options when running this program given as arguments 'close' or 'wide'. The wide angle camera can detect the whole PCB, and the close up camera can detect the hole and wire.

Example commands to run the program: ``` python main.py close ``` or ``` python main.py wide ```
