# VOD_Tools
VOD Tools is a set of utilities for generating analytics from video in a modular and scalable fasion.

The overall workflow of the VOD tool should allow a video to be processed via image frames,
with the goal to generate some timestamped analytics data to be used directly or further processed in data science tools.

![High level block diagram of video analysis workflow](/docs/Overview.png)

Typically the image processing is not linear: There can be multiple unrelated processes which could be performed in parallel and have results rejoin later on.
In some cases image processing steps result in optimizations for more compute-heavy stages (eg limit processing if no motion is occurring).
Multiple outputs for the image pipeline is also possible: If we want to see visualizations from any of the pipeline stages (for presentation or debugging)
it should be easy for a developer to plug in a visualizer that uses the output of a stage to generate a video stream.
One potential real-world workflow is shown below: An object tracking algorithm is used to generate analytics with the help of a motion compensation algorithm.
In order to show how the motion compensation is operating, a visualization of the optical flow calculation is also dumped to a video file.

![Example of workflow requiring the use of object tracking](/docs/Workflows.png)

# Application
An example application is to create analytics on the video game Valorant, a First-Person Shooter developed by Riot Games.
One goal of the game is to eliminate the opposing team which requires good reaction time and accuracy in a gunfight.

In order to train these skills, guides have been posted online on ways to improve game performance.
One such is [Woohoojin’s Youtube guide](https://www.youtube.com/watch?v=q6qv17jgLY4), hereafter referred to as “Overaim drill”.
Let's use videos of the overaim drill to track key metrics outlined in the guide and plot a player's improvement over time.

If you are interested in looking under the hood, read on below in section [Overaim Drill](#overaim-drill)

# How to use
```
python video.py <filename> –options
```
Options:
>`–dump <filename> / -d: Generate a video output based on the debug output frame (first if there are multiple)`

>`–skip <num>: Skip num frames from the start of the video in case you have preamble / setup`

>`–manual / -m: put the video player in manual play mode. It will advance one frame for each keypress`

>`–batch / -b: runs the tool in headless mode without a display. It will automatically run through the video regardless of -m`

Ensure that your python environment / venv has all the packages listed in requirements.txt.
It should work in Python 3.10 and 3.11

In the video player, you can use the following keys:
- k = play or pause the video
- r/f = flip through different debug views (if there are multiple)
- q = exit the program
- any other key to advance by one frame

# Modules
Image processing modules can be designed and tested individually and integrated into an overall pipeline.
`ImgTask.ImgTask` is a base class which can be extended to create a new step in the pipeline.
Let's look at an example of how to create a module that writes video output (found in ImgProc/VideoWriter.py):

```
class VideoWriter(ImgTask.ImgTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = 'dump.mp4'

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        frate = kwargs[ImgTask._FR_KEY]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.sink = cv2.VideoWriter(self.path, fourcc, frate, (self.xdim, self.ydim))
        
    def close(self):
        if self.sink is not None:
            self.sink.release()
            
    def requires(self):
        return [ImgTask.VAL_FRAMENUM, ImgTask.VAL_TS, ImgTask.IMG_DEBUG]
        
    def outputs(self):
        return [WRITER_NODE]

    def proc_frame(self, frame_num, ts, frame):
        img = self.filter_source(frame)
        draw_text(img, '%6d'%(frame_num), self.xdim-150,45)
        draw_text(img, '%6d'%(ts), self.xdim-150,75)
        self.sink.write(img)
        return 0
        
    def filter_source(self, frame):
        if isinstance(frame, dict):
            frame = list(frame.values())[0]
        elif isinstance(frame, list):
            frame = frame[0]
        return np.array(frame)
```

At the bare minimum, a module should implement the "requires", "outputs", and "proc_frame" function.
Let's break down each function:

`def __init__(self, **kwargs):`
The class constructor can be used to set up any variables or functionality which is required prior to initialization.

`def initialize(self, **kwargs):`
The initialize function is called by a global pubsub when the video stream has been set up. At this point parameters such as frame dimension,
frame rate and pipeline registration is completed. If module specific setup that requires video metadata is necessary, this is a good spot to perform them.
Note at this point the superclass `initialize` has registered the module as a task, so "requires" and "outputs" can no longer be modified.


Native to ImgTask, the following variables are available:

>self.xdim = Width of the video source in pixels

>self.ydim = Height of the video source in pixels

>self.midx = X-coordinate where the middle of the scene is located (Width/2)

>self.midy = Y-coordinate where the middle of the scene is located (Height/2)

>self.depth = Depth of the video source (color channels: 3 for RGB, 1 for grayscale)

For VideoWriter, the output stream is opened at this point with parameters matching the source video

`def close(self):`
If there are any clean up operations that need to be performed at the end of video, the global pubsub will call this function.

For VideoWriter, the stream needs to be closed out properly to avoid corruption.

`def requires(self):`
This is a list of inputs required for the module to complete successfully. Each pipeline node requires a unique mnemonic and some common ones are listed in ImgProc/ImgTask.py.
Any string can be used as long as there is also an "output" in the pipeline which supplies the requirement.
If not requirements are present, the pipeline scheduler may skip execution of the module.

Note that list of "requires" must match the order of arguments in "proc_frame". Argument names are irrelevant.

For VideoWriter, it uses the frame number and timestamp in order to augment the output image and takes the debug output as the frame source

`def outputs(self):`
This is a list of outputs from the module. Each pipeline node requires a unique mnemonic and some common ones are listed in ImgProc/ImgTask.py.
Any string can be used and will be used to feed downstream modules that list the same string as "requires".
Note that "proc_frame" must return the same number of outputs (tuple) as listed in "outputs"

For VideoWriter, there is no downstream node. However to ensure that the writer is executed, a custom node is created and referenced in the main pipeline output requirements.

`def proc_frame(self, frame_num, ts, frame):`
The meat of the module, this function is called for every frame coming through the pipeline. See notes on "requires" and "outputs" regarding signature matching

For VideoWriter, it parses out the debug output and takes the first frame (if there are multiple debugs) and augments frame numbering data in the corner before saving to stream.

## Auto registration
At the end of each module, if an instance of the class is created then it will automatically hook into the pubsub system to be registered/called.

Eg. in VideoWriter: `_ = VideoWriter()`

This allows the main program to be streamlined in not having to deal with creation of objects and initialization.
Simply import your module and it will be automatically included in the compute graph.
If a module is not required to generate the final output, the pipeline scheduler will skip its execution.

# Overaim Drill
Here is a breakdown of how to compose a workflow for the overaim drill. For reference, below is an image of the compute graph generated by the pipeline.

![Overaim drill workflow](/docs/Rangeflow.png)

In order to generate the metrics described in the overaim drill guide, a few key parameters are required:
1. Video timing data
2. Keyboard/Mouse inputs
3. Bot appearance, location
4. Mouse motion / proxy

In addition to getting these values, the datapoints will also need to be tied into a coherent sequence of events.
A standard sequences of events appears as follows:

Bot appear &rarr; mouse/keyboard start &rarr; mouse past bot head (overaim) &rarr; micro-adjustment &rarr; mouse/keyboard stop &rarr; target confirmation &rarr; shoot &rarr; bot disappear

### Video timing data
Starting with the simplest, video timing data is required to calculate many of the time metrics (reaction time, flick time, etc) and be portable across different streams (30fps vs 60fps).
Fortunately in the OpenCV2 library there is the capability to extract the current timestamp which can also robustly handle a variety of edge cases such as frame skips, variable timebases or numerical accuracy.
Theoretically it can also be calculated from the frame count if the framerate stability is guaranteed.

### Keyboard/Mouse inputs
Input overlays are available for common video capture software such as OBS, and activity on these overlays can be detected.
Some thresholding may be required as the overlays have a degree of transparency, but are otherwise simple.
The "input-overlay" plugin for OBS has yellow highlight when a key is depressed. An example is shown below:

![Example of input overlay](/docs/overlay.png)

The same base module can be used with two instances to detect keyboard action and mouse action. (**InputAnalyzer**)
These two input devices need to be separated to distinguish between shooting (mouse click) and movement (keyboard)

### Bot appearance, location
Bots are the main targets in the overaim drill. In order to score a hit, the player must position the crosshair over the bot's head and shoot.
For a player to be successful, hitting body shots or multiple shots is subpar. Thus only first shot aimed at the head counts.
Calculating reaction time is also dependent on when a player would reasonably be able to identify a target.
As such we need to locate a bot's head with high accuracy, both in terms of location on screen as well as robustly in the presence of obstacles.
There may be cases where a bot is partially obscured by objects (such as the player's gun) but humans have no problem extrapolating the head position.

To accomplish this robustly, a Deep Neural Net approach is used leveraging the YOLO library.
While the pre-trained object detector can identify human bodies with good accuracy, the bots in the game do not resemble the people in the training dataset.
Unnatural skin tones, glowing outlines, the lack of facial features result in unreliable detections (example image below).
As such the existing pose-estimation model on the COCO dataset needs to be custom-tuned to work well with the game bots. (**BotFind**)

![Example of bot in game](/docs/bot.png)

A pose estimation model is used here because there are situations in which the head may be obscured.
Humans can typically extrapolate the position of someone's head given enough visibility of their body / other skeletal features, where as a pure head object detection model would not be able to do so.
Pose estimation can be more robust in determining the scale of the bot by using auxiliary features such as shoulder width / body length.

As Machine Learning models can be computationally expensive to run, it is best to minimize the amount of processing that needs to be done.
Some of these optimizations are as follows:
- Process only areas of the frame where pixel changes have occurred
- If we are already tracking a bot, we should be able to predict its new location using motion flow algorithms
- In cases where the motion flow is inaccurate, the bot should still be in a nearby vicinity (rather than searching the entire view)

To support these operations, modules used to calculate standard Computer Vision algorithms have been implemented (**Delta, OpticFlow, BotTrack**).

Finally, event data needs to be generated when the bot of interest appears, disappears or is in motion (**BestBot, PoseAnalyzer**). 

### Mouse motion
Mouse motion is not as straightforward as detecting key inputs. Overlays showing mouse movement will show a general direction but have no fine 2D information, which is critical for distinguishing types of motion.
There are two approaches to extract mouse data:
1. Use the bot head location as the "inverse" mouse motion
2. Use Motion Flow algorithms to estimate screen trajectory

While using the bot head location would result in a simpler algorithm, there are problems with this approach:
Situations in which no bots are present would render this implementation unusable, and in cases where the bot is in motion or numerical accuracy jittering of ML localization can throw off analysis.

As such it is better to base the tracking off Motion Flow algorithms. Best case is reusing the same modules from bot tracking to save on computation.
Due to details of camera and game physics, a true motion flow estimator to calculate mouse movement becomes a multi-variate regression problem.
To make an estimator which is more performant, let's boil down a dense optic flow into a single vector which represents translation of the entire frame, which can be accomplished with a least-squares / RANSAC fitting.
This movement vector does not represent the mouse input exactly, but a blend of camera panning, body translation, and the assumption that the player is aiming at the horizon. 
More importantly there is good data about the magnitude and direction of motion which can be fed into a state machine to distinguish the player's intent:
1. Not moving (idle, waiting to react)
2. Coarse motion (fast flick)
3. Fine motion (micro-adjustment)
4. Target confirmation

By breaking down mouse motion into these four stages, it is possible to generate events around them and feed it into the analytics. (**MoveStage**)

## Data Analytics
Typically a run of the overaim drill involves 30 bots appearing in sequence at random positions on the screen. 
Running the pipeline to generate analytics for each trial, we can calculate the following:
1. Reaction time from bot appearing to first action (mouse or keyboard)
2. Reaction time to key press
3. Reaction time to mouse movement
4. Time to flick past bot head (overaim)
5. Landing position of overaim (distance away from bot head)
6. How long a player spends moving
7. Time spent micro-adjusting
8. Time spent confirming aim
9. Time from bot appear to shot fired
10. Position of shot (from bot head)
11. Initial position/scale of bot (from crosshair)

ISSUE-2 in the guide is addressed by points 1, 2, 3.  
ISSUE-3 in the guide is addressed by points 5, 7, 8.  
ISSUE-4 in the guide is addressed by points 4, 5.  
ISSUE-6 in the guide is addressed by points 5, 6.  

In addition to the median value, the variance of the trials can help identify issues regarding consistency.
The last metrics can be useful for training. Points 9, 10 show overall performance, and 11 acts as a proxy for trial difficulty.
 
## Current limitations
Bot detection model is trained using data collected by the author's existing training runs, with 1920x1080x60 video and yellow bot outlines on medium difficulty.
Detection accuracy on other settings may be degraded.