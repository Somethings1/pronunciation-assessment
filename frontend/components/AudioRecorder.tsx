import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square } from 'lucide-react';

interface AudioRecorderProps {
  onRecordingComplete: (blob: Blob) => void;
  isProcessing: boolean;
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onRecordingComplete, isProcessing }) => {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Initialize with null to prevent TS error: Expected 1 arguments, but got 0.
  const animationRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (audioContextRef.current) audioContextRef.current.close();
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      // Audio Viz Setup
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;
      const analyser = audioContext.createAnalyser();
      analyserRef.current = analyser;
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      analyser.fftSize = 256;

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
        onRecordingComplete(blob);
        stream.getTracks().forEach(track => track.stop()); // Stop mic
        if (animationRef.current) cancelAnimationFrame(animationRef.current);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      drawVisualizer();
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Could not access microphone. Please allow permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const drawVisualizer = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!analyserRef.current) return;
      animationRef.current = requestAnimationFrame(draw);
      analyserRef.current.getByteFrequencyData(dataArray);

      ctx.fillStyle = 'rgb(248, 250, 252)'; // slate-50
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.5;
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        barHeight = dataArray[i] / 2;
        // Gradient color
        ctx.fillStyle = `rgb(${barHeight + 100}, 99, 132)`;
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        x += barWidth + 1;
      }
    };
    draw();
  };

  return (
    <div className="flex flex-col items-center justify-center space-y-6">
      <div className="relative w-full max-w-md h-24 bg-slate-100 rounded-xl overflow-hidden border border-slate-200">
        <canvas 
          ref={canvasRef} 
          width={400} 
          height={100} 
          className="w-full h-full"
        />
        {!isRecording && !isProcessing && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">
            Click microphone to start
          </div>
        )}
        {isProcessing && (
           <div className="absolute inset-0 flex items-center justify-center bg-slate-50/80 text-indigo-600 font-semibold animate-pulse">
            Analyzing Audio...
          </div>
        )}
      </div>

      <button
        onClick={isRecording ? stopRecording : startRecording}
        disabled={isProcessing}
        className={`
          relative flex items-center justify-center w-20 h-20 rounded-full shadow-lg transition-all duration-200
          ${isProcessing ? 'bg-slate-300 cursor-not-allowed' : 
            isRecording 
              ? 'bg-red-500 hover:bg-red-600 animate-pulse-slow ring-4 ring-red-200' 
              : 'bg-indigo-600 hover:bg-indigo-700 hover:scale-105 ring-4 ring-indigo-100'}
        `}
      >
        {isRecording ? (
          <Square className="w-8 h-8 text-white fill-current" />
        ) : (
          <Mic className="w-8 h-8 text-white" />
        )}
      </button>
      <p className="text-slate-500 text-sm font-medium">
        {isRecording ? "Listening..." : "Tap to Record"}
      </p>
    </div>
  );
};

export default AudioRecorder;