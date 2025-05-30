#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Fri May 30 14:25:18 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'N170'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1512, 982]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = ('sub-'+ expInfo['participant'] + expName + '_raw')
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/ellepedaci/Desktop/Experiments/PA3/N170/N170.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    from psychopy.hardware import joystick as joysticklib  # joystick/gamepad accsss
    from psychopy.experiment.components.joyButtons import virtualJoyButtons as virtualjoybuttonslib
    from psychopy.hardware import joystick as joysticklib  # joystick/gamepad accsss
    from psychopy.experiment.components.joyButtons import virtualJoyButtons as virtualjoybuttonslib
    from psychopy.hardware import joystick as joysticklib  # joystick/gamepad accsss
    from psychopy.experiment.components.joyButtons import virtualJoyButtons as virtualjoybuttonslib
    
    # --- Initialize components for Routine "instructions" ---
    text_instructions = visual.TextStim(win=win, name='text_instructions',
        text='In this experiment, you will be shown a series of images. Your task is to press the B button when you see an image that you have previously seen.\n\nPress any button to begin.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button_resp_inst = type('', (), {})() # Create an object to use as a name space
    button_resp_inst.device = None
    button_resp_inst.device_number = 0
    
    try:
        numJoysticks = joysticklib.getNumJoysticks()
        if numJoysticks > 0:
            button_resp_inst.device = joysticklib.Joystick(0)
            try:
                joystickCache
            except NameError:
                joystickCache={}
            if not 0 in joystickCache:
                joystickCache[0] = joysticklib.Joystick(0)
            button_resp_inst.device = joystickCache[0]
        else:
            button_resp_inst.device = virtualjoybuttonslib.VirtualJoyButtons(0)
            logging.warning("joystick_{}: Using keyboard emulation 'ctrl' + 'Alt' + digit.".format(button_resp_inst.device_number))
    except Exception:
        pass
        
    if not button_resp_inst.device:
        logging.error('No joystick/gamepad device found.')
        core.quit()
    
    button_resp_inst.status = None
    button_resp_inst.clock = core.Clock()
    button_resp_inst.numButtons = button_resp_inst.device.getNumButtons()
    
    
    # --- Initialize components for Routine "image_stim" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    button_resp = type('', (), {})() # Create an object to use as a name space
    button_resp.device = None
    button_resp.device_number = 0
    
    try:
        numJoysticks = joysticklib.getNumJoysticks()
        if numJoysticks > 0:
            button_resp.device = joysticklib.Joystick(0)
            try:
                joystickCache
            except NameError:
                joystickCache={}
            if not 0 in joystickCache:
                joystickCache[0] = joysticklib.Joystick(0)
            button_resp.device = joystickCache[0]
        else:
            button_resp.device = virtualjoybuttonslib.VirtualJoyButtons(0)
            logging.warning("joystick_{}: Using keyboard emulation 'ctrl' + 'Alt' + digit.".format(button_resp.device_number))
    except Exception:
        pass
        
    if not button_resp.device:
        logging.error('No joystick/gamepad device found.')
        core.quit()
    
    button_resp.status = None
    button_resp.clock = core.Clock()
    button_resp.numButtons = button_resp.device.getNumButtons()
    
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "end" ---
    text_end = visual.TextStim(win=win, name='text_end',
        text='That concludes the experiment.\n\nPress any button to finish the session. \n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button_resp_end = type('', (), {})() # Create an object to use as a name space
    button_resp_end.device = None
    button_resp_end.device_number = 0
    
    try:
        numJoysticks = joysticklib.getNumJoysticks()
        if numJoysticks > 0:
            button_resp_end.device = joysticklib.Joystick(0)
            try:
                joystickCache
            except NameError:
                joystickCache={}
            if not 0 in joystickCache:
                joystickCache[0] = joysticklib.Joystick(0)
            button_resp_end.device = joystickCache[0]
        else:
            button_resp_end.device = virtualjoybuttonslib.VirtualJoyButtons(0)
            logging.warning("joystick_{}: Using keyboard emulation 'ctrl' + 'Alt' + digit.".format(button_resp_end.device_number))
    except Exception:
        pass
        
    if not button_resp_end.device:
        logging.error('No joystick/gamepad device found.')
        core.quit()
    
    button_resp_end.status = None
    button_resp_end.clock = core.Clock()
    button_resp_end.numButtons = button_resp_end.device.getNumButtons()
    
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[text_instructions, button_resp_inst],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    button_resp_inst.oldButtonState = button_resp_inst.device.getAllButtons()[:]
    button_resp_inst.keys = []
    button_resp_inst.rt = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions* updates
        
        # if text_instructions is starting this frame...
        if text_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions.frameNStart = frameN  # exact frame index
            text_instructions.tStart = t  # local t and not account for scr refresh
            text_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instructions.started')
            # update status
            text_instructions.status = STARTED
            text_instructions.setAutoDraw(True)
        
        # if text_instructions is active this frame...
        if text_instructions.status == STARTED:
            # update params
            pass
        
        # *button_resp_inst* updates
        
        # if button_resp_inst is starting this frame...
        if button_resp_inst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            button_resp_inst.frameNStart = frameN  # exact frame index
            button_resp_inst.tStart = t  # local t and not account for scr refresh
            button_resp_inst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_resp_inst, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_resp_inst.started')
            # update status
            button_resp_inst.status = STARTED
            # joyButtons checking is just starting
            win.callOnFlip(button_resp_inst.clock.reset)  # t=0 on next screen flip
        if button_resp_inst.status == STARTED:
            button_resp_inst.newButtonState = button_resp_inst.device.getAllButtons()[:]
            button_resp_inst.pressedButtons = []
            button_resp_inst.releasedButtons = []
            button_resp_inst.newPressedButtons = []
            if button_resp_inst.newButtonState != button_resp_inst.oldButtonState:
                button_resp_inst.pressedButtons = [i for i in range(button_resp_inst.numButtons) if button_resp_inst.newButtonState[i] and not button_resp_inst.oldButtonState[i]]
                button_resp_inst.releasedButtons = [i for i in range(button_resp_inst.numButtons) if not button_resp_inst.newButtonState[i] and button_resp_inst.oldButtonState[i]]
                button_resp_inst.oldButtonState = button_resp_inst.newButtonState
                button_resp_inst.newPressedButtons = [i for i in range(button_resp_inst.numButtons) if i in button_resp_inst.pressedButtons]
                [logging.data("joystick_{}_button: {}".format(button_resp_inst.device_number,i)) for i in button_resp_inst.pressedButtons]
            theseKeys = button_resp_inst.newPressedButtons
            if len(theseKeys) > 0:  # at least one key was pressed
                button_resp_inst.keys = theseKeys[-1]  # just the last key pressed
                button_resp_inst.rt = button_resp_inst.clock.getTime()
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # check responses
    if button_resp_inst.keys in ['', [], None]:  # No response was made
        button_resp_inst.keys=None
    thisExp.addData('button_resp_inst.keys',button_resp_inst.keys)
    if button_resp_inst.keys != None:  # we had a response
        thisExp.addData('button_resp_inst.rt', button_resp_inst.rt)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    image_loop = data.TrialHandler2(
        name='image_loop',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('N170stimulus_lists/sub-' + expInfo['participant']+'_N170_stimlist.csv'), 
        seed=None, 
    )
    thisExp.addLoop(image_loop)  # add the loop to the experiment
    thisImage_loop = image_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisImage_loop.rgb)
    if thisImage_loop != None:
        for paramName in thisImage_loop:
            globals()[paramName] = thisImage_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisImage_loop in image_loop:
        currentLoop = image_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisImage_loop.rgb)
        if thisImage_loop != None:
            for paramName in thisImage_loop:
                globals()[paramName] = thisImage_loop[paramName]
        
        # --- Prepare to start Routine "image_stim" ---
        # create an object to store info about Routine image_stim
        image_stim = data.Routine(
            name='image_stim',
            components=[image, button_resp, text],
        )
        image_stim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image.setImage(name)
        button_resp.oldButtonState = button_resp.device.getAllButtons()[:]
        button_resp.keys = []
        button_resp.rt = []
        # Run 'Begin Routine' code from code_ISI
        isi_len = np.random.uniform(0.75, 1.25, 1)
        button_len = isi_len + 0.3
        text.setText('+')
        # store start times for image_stim
        image_stim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        image_stim.tStart = globalClock.getTime(format='float')
        image_stim.status = STARTED
        thisExp.addData('image_stim.started', image_stim.tStart)
        image_stim.maxDuration = None
        # keep track of which components have finished
        image_stimComponents = image_stim.components
        for thisComponent in image_stim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "image_stim" ---
        # if trial has changed, end Routine now
        if isinstance(image_loop, data.TrialHandler2) and thisImage_loop.thisN != image_loop.thisTrial.thisN:
            continueRoutine = False
        image_stim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # if image is stopping this frame...
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + .3-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.tStopRefresh = tThisFlipGlobal  # on global time
                    image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.stopped')
                    # update status
                    image.status = FINISHED
                    image.setAutoDraw(False)
            
            # *button_resp* updates
            
            # if button_resp is starting this frame...
            if button_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                button_resp.frameNStart = frameN  # exact frame index
                button_resp.tStart = t  # local t and not account for scr refresh
                button_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_resp.started')
                # update status
                button_resp.status = STARTED
                # joyButtons checking is just starting
                win.callOnFlip(button_resp.clock.reset)  # t=0 on next screen flip
            
            # if button_resp is stopping this frame...
            if button_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > button_resp.tStartRefresh + button_len-frameTolerance:
                    # keep track of stop time/frame for later
                    button_resp.tStop = t  # not accounting for scr refresh
                    button_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    button_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'button_resp.stopped')
                    # update status
                    button_resp.status = FINISHED
            if button_resp.status == STARTED:
                button_resp.newButtonState = button_resp.device.getAllButtons()[:]
                button_resp.pressedButtons = []
                button_resp.releasedButtons = []
                button_resp.newPressedButtons = []
                if button_resp.newButtonState != button_resp.oldButtonState:
                    button_resp.pressedButtons = [i for i in range(button_resp.numButtons) if button_resp.newButtonState[i] and not button_resp.oldButtonState[i]]
                    button_resp.releasedButtons = [i for i in range(button_resp.numButtons) if not button_resp.newButtonState[i] and button_resp.oldButtonState[i]]
                    button_resp.oldButtonState = button_resp.newButtonState
                    button_resp.newPressedButtons = [i for i in [2] if i in button_resp.pressedButtons]
                    [logging.data("joystick_{}_button: {}".format(button_resp.device_number,i)) for i in button_resp.pressedButtons]
                theseKeys = button_resp.newPressedButtons
                if len(theseKeys) > 0:  # at least one key was pressed
                    button_resp.keys = theseKeys[-1]  # just the last key pressed
                    button_resp.rt = button_resp.clock.getTime()
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= .3-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + isi_len-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                image_stim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in image_stim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "image_stim" ---
        for thisComponent in image_stim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for image_stim
        image_stim.tStop = globalClock.getTime(format='float')
        image_stim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('image_stim.stopped', image_stim.tStop)
        # check responses
        if button_resp.keys in ['', [], None]:  # No response was made
            button_resp.keys=None
        image_loop.addData('button_resp.keys',button_resp.keys)
        if button_resp.keys != None:  # we had a response
            image_loop.addData('button_resp.rt', button_resp.rt)
        # the Routine "image_stim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'image_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[text_end, button_resp_end],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    button_resp_end.oldButtonState = button_resp_end.device.getAllButtons()[:]
    button_resp_end.keys = []
    button_resp_end.rt = []
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end* updates
        
        # if text_end is starting this frame...
        if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end.frameNStart = frameN  # exact frame index
            text_end.tStart = t  # local t and not account for scr refresh
            text_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end.started')
            # update status
            text_end.status = STARTED
            text_end.setAutoDraw(True)
        
        # if text_end is active this frame...
        if text_end.status == STARTED:
            # update params
            pass
        
        # if text_end is stopping this frame...
        if text_end.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_end.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                text_end.tStop = t  # not accounting for scr refresh
                text_end.tStopRefresh = tThisFlipGlobal  # on global time
                text_end.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_end.stopped')
                # update status
                text_end.status = FINISHED
                text_end.setAutoDraw(False)
        
        # *button_resp_end* updates
        
        # if button_resp_end is starting this frame...
        if button_resp_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            button_resp_end.frameNStart = frameN  # exact frame index
            button_resp_end.tStart = t  # local t and not account for scr refresh
            button_resp_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_resp_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_resp_end.started')
            # update status
            button_resp_end.status = STARTED
            # joyButtons checking is just starting
            win.callOnFlip(button_resp_end.clock.reset)  # t=0 on next screen flip
        
        # if button_resp_end is stopping this frame...
        if button_resp_end.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > button_resp_end.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                button_resp_end.tStop = t  # not accounting for scr refresh
                button_resp_end.tStopRefresh = tThisFlipGlobal  # on global time
                button_resp_end.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_resp_end.stopped')
                # update status
                button_resp_end.status = FINISHED
        if button_resp_end.status == STARTED:
            button_resp_end.newButtonState = button_resp_end.device.getAllButtons()[:]
            button_resp_end.pressedButtons = []
            button_resp_end.releasedButtons = []
            button_resp_end.newPressedButtons = []
            if button_resp_end.newButtonState != button_resp_end.oldButtonState:
                button_resp_end.pressedButtons = [i for i in range(button_resp_end.numButtons) if button_resp_end.newButtonState[i] and not button_resp_end.oldButtonState[i]]
                button_resp_end.releasedButtons = [i for i in range(button_resp_end.numButtons) if not button_resp_end.newButtonState[i] and button_resp_end.oldButtonState[i]]
                button_resp_end.oldButtonState = button_resp_end.newButtonState
                button_resp_end.newPressedButtons = [i for i in range(button_resp_end.numButtons) if i in button_resp_end.pressedButtons]
                [logging.data("joystick_{}_button: {}".format(button_resp_end.device_number,i)) for i in button_resp_end.pressedButtons]
            theseKeys = button_resp_end.newPressedButtons
            if len(theseKeys) > 0:  # at least one key was pressed
                button_resp_end.keys = theseKeys[-1]  # just the last key pressed
                button_resp_end.rt = button_resp_end.clock.getTime()
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # check responses
    if button_resp_end.keys in ['', [], None]:  # No response was made
        button_resp_end.keys=None
    thisExp.addData('button_resp_end.keys',button_resp_end.keys)
    if button_resp_end.keys != None:  # we had a response
        thisExp.addData('button_resp_end.rt', button_resp_end.rt)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end.maxDurationReached:
        routineTimer.addTime(-end.maxDuration)
    elif end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
