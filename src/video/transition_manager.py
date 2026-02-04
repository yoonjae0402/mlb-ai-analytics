
from moviepy.editor import VideoClip, CompositeVideoClip, vfx, concatenate_videoclips
from typing import List

class TransitionManager:
    """
    Manages smooth transitions between scenes (Crossfade, Wipes).
    """

    @staticmethod
    def crossfade(clip1: VideoClip, clip2: VideoClip, duration: float = 0.5) -> VideoClip:
        """
        Crossfades outgoing clip1 with incoming clip2.
        Outcome is a single composite clip.
        """
        # Apply padding or ensure overlap
        # Standard: clip1 fades out, clip2 fades in, they overlap by duration
        
        clip1_faded = clip1.fx(vfx.fadeout, duration)
        clip2_faded = clip2.fx(vfx.fadein, duration)
        
        # Overlap requires specific Composite timing
        # Start clip2 at (clip1.duration - duration)
        
        start_time_c2 = clip1.duration - duration
        clip2_faded = clip2_faded.set_start(start_time_c2)
        
        return CompositeVideoClip([clip1_faded, clip2_faded]).set_duration(start_time_c2 + clip2.duration)

    @staticmethod
    def apply_transitions(clips: List[VideoClip], transition_type: str = 'crossfade') -> VideoClip:
        """
        Chains a list of clips with transitions using a flat CompositeVideoClip
        to avoid recursion depth issues and improve performance.
        """
        if not clips:
            return None
        if len(clips) == 1:
            return clips[0]

        transition_dur = 0.5 # Duration of overlap
        
        # Method: Layout all clips on a timeline
        # Clip 0: Start 0, Dur D0
        # Clip 1: Start D0 - Trans, Dur D1
        # Clip 2: Start (Start1 + D1) - Trans, ...
        
        final_clips_to_compose = []
        current_start = 0.0
        
        for i, clip in enumerate(clips):
            # Apply fade in/out to the clip itself if it's involved in a transition
            # For strict crossfade:
            # - Outgoing clip fades out
            # - Incoming clip fades in
            
            # We need to process the clip to add fades BEFORE setting start time
            # But wait, 'clips' passed here already might have effects. 
            # Ideally we apply fades here.
            
            processed_clip = clip
            
            # Fade In (except first)
            if i > 0:
                processed_clip = processed_clip.fx(vfx.fadein, transition_dur)
            
            # Fade Out (except last)
            if i < len(clips) - 1:
                processed_clip = processed_clip.fx(vfx.fadeout, transition_dur)
            
            # Set timing
            processed_clip = processed_clip.set_start(current_start)
            final_clips_to_compose.append(processed_clip)
            
            # Advance time (subtracting overlap)
            current_start += (clip.duration - transition_dur)
            
        # Add the duration of the last transition back? 
        # Logic: 
        # Start0 = 0. End0 = D0.
        # Start1 = D0 - 0.5. End1 = Start1 + D1.
        # Start2 = End1 - 0.5.
        
        # Total duration is roughly Sum(Durations) - (N-1)*Overlap
        
        return CompositeVideoClip(final_clips_to_compose)
