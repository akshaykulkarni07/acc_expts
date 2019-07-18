### Re-annotation of Tower Robogame 2.0 IMU data

The data collected by previous Ph.D. student Ewerton Lopes Oliviera has some
issues. The data has 50% overlap and then having different labels for
consecutive examples becomes questionable. The possible mistake here being 
that labelling was done after segmentation (and not the other way round).

Thus we need to re-annotate the data following this : first annotate
the entire data in a continuous manner and then break into segments (all readings
in a segment will have same label).

Original Classes : Sprinting, Moving, Dodging, Blocking, Inactive
New Classes : Running, Walking, Blocking, Inactive, Transient

General rules followed during re-annotation are listed below :
1. Label activities continuously i.e. from start time to end time of one
particular activity being performed, the label will remain the same
(irrespective of time for which the activity is performed).
2. Rule 1 indicates that annotation should not be done in a segmented manner.
For example, don't segment into intervals of (say) 2 seconds and then annotate
each 2-second interval with an appropriate label.
3. Any activity that cannot be properly classified into the classes should be
labeled 'Transient'. For example, if the player is moving fast, but not fast
enough for running and also not slow enough for walking.
4. If there are any frame skips in the video (i.e. frames that were lost in
transmission), then all the IMU readings after that point should be 
labeled 'Transient' since they are not synchronized with the video. If not
done, it could potentially be mislabelling of examples, which would not be
good for the training algorithms.
