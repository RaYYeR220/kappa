// A small controller around the workspace <video>: frame-accurate seeking, the
// current frame as an external store (so only the readouts re-render per frame).

type Listener = () => void;

export class Player {
  video: HTMLVideoElement | null = null;
  private frame = 0;
  private playing = false;
  private rate = 1;
  private listeners = new Set<Listener>();

  constructor(
    readonly fps: number,
    readonly nFrames: number,
  ) {}

  subscribe = (fn: Listener) => {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  };

  getFrame = () => this.frame;
  isPlaying = () => this.playing;
  getRate = () => this.rate;

  private emit() {
    for (const fn of this.listeners) fn();
  }

  attach(video: HTMLVideoElement | null) {
    this.video = video;
  }

  /** The decoder's presented frame reports its timestamp; frame k shows from k / fps. */
  frameAt(mediaTime: number): number {
    return Math.max(0, Math.min(this.nFrames - 1, Math.round(mediaTime * this.fps)));
  }

  setFrame(frame: number) {
    if (frame !== this.frame) {
      this.frame = frame;
      this.emit();
    }
  }

  setPlaying(playing: boolean) {
    if (playing !== this.playing) {
      this.playing = playing;
      this.emit();
    }
  }

  seek(frame: number) {
    const f = Math.max(0, Math.min(this.nFrames - 1, Math.round(frame)));
    this.setFrame(f);
    if (this.video) this.video.currentTime = (f + 0.5) / this.fps;
  }

  step(delta: number) {
    this.pause();
    this.seek(this.frame + delta);
  }

  play() {
    const v = this.video;
    if (!v) return;
    if (this.frame >= this.nFrames - 1) this.seek(0);
    void v.play().catch(() => this.setPlaying(false));
  }

  pause() {
    this.video?.pause();
  }

  toggle() {
    if (this.playing) this.pause();
    else this.play();
  }

  setRate(rate: number) {
    this.rate = rate;
    if (this.video) this.video.playbackRate = rate;
    this.emit();
  }
}
