import React, { useState, useEffect } from 'react';
import { AppState, DifficultyBand, WordData, AssessmentResult } from './types';
import { getRandomWord, getWordsByBand, WORD_DATABASE } from './services/wordService';
import { assessPronunciation } from './services/geminiService'; 
import AudioRecorder from './components/AudioRecorder';
import ResultTable from './components/ResultTable';
import { Volume2, Trophy, ArrowRight, RotateCcw } from './components/Icons';

const App: React.FC = () => {
  const [appState, setAppState] = useState<AppState>('MENU');
  const [selectedBand, setSelectedBand] = useState<DifficultyBand | null>(null);
  const [currentWord, setCurrentWord] = useState<WordData | null>(null);
  const [passedWords, setPassedWords] = useState<string[]>([]); // Array of Word IDs
  const [assessmentResult, setAssessmentResult] = useState<AssessmentResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Speak the current word using Browser TTS
  const playReferenceAudio = () => {
    if (!currentWord) return;
    const utterance = new SpeechSynthesisUtterance(currentWord.word);
    utterance.lang = 'en-US';
    utterance.rate = 0.8; // Slightly slower for learning
    window.speechSynthesis.speak(utterance);
  };

  // Start Practice Session
  const startSession = (band: DifficultyBand) => {
    setSelectedBand(band);
    const word = getRandomWord(band, passedWords);
    if (word) {
      setCurrentWord(word);
      setAppState('PRACTICE');
      setAssessmentResult(null);
    } else {
      alert("You've completed all words in this band!");
    }
  };

  // Handle Next Word (Skip or After Success)
  const handleNextWord = () => {
    if (!selectedBand) return;
    const word = getRandomWord(selectedBand, passedWords);
    if (word) {
      setCurrentWord(word);
      setAssessmentResult(null);
    } else {
      setAppState('MENU');
      alert("Congratulations! Band completed.");
    }
  };

  // Handle Retry
  const handleRetry = () => {
    setAssessmentResult(null);
  };

  // Handle Audio Upload & Assessment
  const handleRecordingComplete = async (blob: Blob) => {
    if (!currentWord) return;
    
    setIsProcessing(true);
    try {
      const result = await assessPronunciation(blob, currentWord);
      setAssessmentResult(result);
      
      // Check Pass Condition: Correct Stress AND Correct Phonemes
      if (result.isPhonemesCorrect && result.isStressCorrect) {
        setPassedWords(prev => {
            if (prev.includes(currentWord.id)) return prev;
            return [...prev, currentWord.id];
        });
      }
    } catch (e) {
      console.error(e);
      alert("Error during assessment. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  // --- RENDER HELPERS ---

  const renderMenu = () => (
    <div className="max-w-4xl mx-auto px-4 py-12 animate-fade-in">
      <div className="text-center mb-12">
        <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 mb-4 tracking-tight">
          Lingo<span className="text-indigo-600">Stress</span>
        </h1>
        <p className="text-lg text-slate-600 max-w-2xl mx-auto">
          Master English pronunciation with AI-powered assessment. 
          Focusing on <span className="font-semibold text-indigo-600">Phoneme Accuracy</span> and <span className="font-semibold text-indigo-600">Syllable Stress</span>.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 md:gap-6">
        {Object.values(DifficultyBand).map((band) => {
            const totalWords = getWordsByBand(band).length;
            const passedInBand = passedWords.filter(id => {
                const w = WORD_DATABASE.find(word => word.id === id);
                return w?.band === band;
            }).length;
            const progressPercent = Math.round((passedInBand / totalWords) * 100);

            return (
              <button
                key={band}
                onClick={() => startSession(band)}
                className="group relative overflow-hidden bg-white border border-slate-200 hover:border-indigo-500 rounded-2xl p-6 shadow-sm hover:shadow-xl transition-all duration-300 flex flex-col items-center"
              >
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-indigo-600 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300"></div>
                <span className="text-4xl font-black text-slate-200 group-hover:text-indigo-100 transition-colors mb-2">{band}</span>
                <span className="text-lg font-bold text-slate-800 group-hover:text-indigo-600">Level {band}</span>
                
                {/* Progress Bar & Stats */}
                <div className="w-full mt-4">
                    <div className="flex justify-between text-xs text-slate-500 mb-1 font-medium">
                        <span>Progress</span>
                        <span>{passedInBand} / {totalWords}</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2">
                        <div 
                            className="bg-emerald-500 h-2 rounded-full transition-all duration-500" 
                            style={{ width: `${progressPercent}%` }}
                        ></div>
                    </div>
                </div>
              </button>
            );
        })}
      </div>
    </div>
  );

  const renderPractice = () => {
    if (!currentWord) return null;

    const isPassed = assessmentResult?.isPhonemesCorrect && assessmentResult?.isStressCorrect;

    return (
      <div className="max-w-3xl mx-auto px-4 py-8 min-h-screen flex flex-col">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <button 
            onClick={() => setAppState('MENU')}
            className="text-slate-500 hover:text-slate-800 font-medium flex items-center text-sm"
          >
            &larr; Back to Bands
          </button>
          <div className="px-3 py-1 bg-indigo-50 text-indigo-700 rounded-full text-xs font-bold tracking-wide uppercase">
            Band {selectedBand}
          </div>
        </div>

        {/* Word Display */}
        <div className="flex-1 flex flex-col items-center">
          
          {/* Success Message - Fixed centered block */}
          {isPassed && (
            <div className="w-full flex justify-center mb-8 animate-bounce">
              <div className="px-8 py-4 bg-emerald-100 text-emerald-900 rounded-full shadow-md border border-emerald-300 flex items-center transform hover:scale-105 transition-transform">
                <Trophy className="w-6 h-6 mr-3 text-emerald-600" />
                <span className="font-extrabold text-lg">Perfect Pronunciation! Word Mastered.</span>
              </div>
            </div>
          )}

          <div className="text-center mb-10">
            <h2 className="text-6xl md:text-7xl font-extrabold text-slate-900 mb-4 break-words">
              {currentWord.word}
            </h2>
            <div className="flex items-center justify-center space-x-4 mb-2">
              <span className="text-2xl text-slate-500 font-mono bg-slate-100 px-3 py-1 rounded-lg">
                {currentWord.ipa}
              </span>
              <button 
                onClick={playReferenceAudio}
                className="p-2 rounded-full bg-indigo-100 text-indigo-600 hover:bg-indigo-200 transition-colors"
                title="Listen to reference"
              >
                <Volume2 className="w-5 h-5" />
              </button>
            </div>
            <p className="text-sm text-slate-400 font-mono tracking-widest mt-2">
              STRESS: {currentWord.stressPattern.split('').map(s => s === '1' ? '●' : '○').join(' ')}
            </p>
          </div>

          {/* Assessment State */}
          {!assessmentResult ? (
            <div className="w-full flex flex-col items-center animate-fade-in-up">
              <AudioRecorder onRecordingComplete={handleRecordingComplete} isProcessing={isProcessing} />
            </div>
          ) : (
            <div className="w-full flex flex-col items-center animate-fade-in-up">
              <ResultTable result={assessmentResult} targetWord={currentWord} />
              
              <div className="flex space-x-4 mt-8 pb-12">
                {!isPassed && (
                  <button
                    onClick={handleRetry}
                    className="flex items-center px-6 py-3 bg-white border border-slate-300 text-slate-700 rounded-lg font-semibold hover:bg-slate-50 transition-colors shadow-sm"
                  >
                    <RotateCcw className="w-4 h-4 mr-2" />
                    Try Again
                  </button>
                )}
                
                <button
                  onClick={handleNextWord}
                  className={`flex items-center px-6 py-3 rounded-lg font-semibold shadow-md transition-all transform hover:scale-105 ${
                    isPassed 
                      ? 'bg-emerald-500 hover:bg-emerald-600 text-white ring-4 ring-emerald-200' 
                      : 'bg-slate-800 hover:bg-slate-900 text-white'
                  }`}
                >
                  {isPassed ? 'Next Word' : 'Skip Word'}
                  <ArrowRight className="w-4 h-4 ml-2" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans selection:bg-indigo-100 selection:text-indigo-800">
      {appState === 'MENU' && renderMenu()}
      {appState === 'PRACTICE' && renderPractice()}
      
      {/* Mock Mode Status Indicator */}
      <div className="fixed bottom-4 right-4 max-w-xs bg-amber-50 border border-amber-200 text-amber-800 p-2 rounded-lg shadow text-[10px] font-mono">
         Mock Mode (Random Results)
      </div>
    </div>
  );
};

export default App;