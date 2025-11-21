import React from 'react';
import { CheckCircle, XCircle, Activity } from 'lucide-react'; // Assuming you use lucide-react or similar
import { AssessmentResult, WordData } from '../types';

interface ResultTableProps {
  result: AssessmentResult;
  targetWord: WordData;
}

const ResultTable: React.FC<ResultTableProps> = ({ result, targetWord }) => {

  // Helper: Determine color based on GOP score (Higher is better, usually negative)
  // Thresholds: > -0.5 (Good), > -2.0 (Fair), <= -2.0 (Poor)
  const getScoreColor = (score: number) => {
    if (score > -0.5) return "bg-emerald-100 text-emerald-800 border-emerald-200";
    if (score > -2.0) return "bg-yellow-100 text-yellow-800 border-yellow-200";
    return "bg-rose-100 text-rose-800 border-rose-200 font-bold";
  };

  // Helper: Render Phoneme blocks
  const renderPhonemes = () => {
    // We iterate over the keys of phonemesScore.
    // Note: The backend keys might be like "AH_0", "P_1" to ensure uniqueness.
    // We sort them by index to maintain order.

    const sortedPhonemes = Object.entries(result.phonemesScore).sort((a, b) => {
      const indexA = parseInt(a[0].split('_').pop() || '0');
      const indexB = parseInt(b[0].split('_').pop() || '0');
      return indexA - indexB;
    });

    return (
      <div className="flex flex-wrap gap-2">
        {sortedPhonemes.map(([key, score]) => {
          const phonemeText = key.split('_')[0]; // "AH_0" -> "AH"
          const colorClass = getScoreColor(score);

          return (
            <div key={key} className="flex flex-col items-center">
              <span className={`px-3 py-1.5 rounded-lg border text-sm font-mono font-bold shadow-sm transition-colors ${colorClass}`}>
                {phonemeText}
              </span>
              <span className="text-[10px] text-slate-400 mt-1 font-mono">
                {Math.round(score * 10) / 10}
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  // Helper: Render Stress Bars
  const renderStressBars = (pattern: number[], label: string) => {
    return (
      <div className="flex flex-col items-center">
        <span className="text-[10px] font-bold text-slate-400 mb-2 uppercase tracking-wider">{label}</span>
        <div className="flex items-end h-16 space-x-2 bg-slate-50 p-2 rounded-lg border border-slate-100">
          {pattern.map((val, i) => {
            const isPrimary = val === 1; // 1 is stressed, 0 is unstressed

            // Visuals
            const height = isPrimary ? 'h-10' : 'h-4';
            const width = 'w-4';
            const opacity = isPrimary ? 'opacity-100' : 'opacity-40';

            // Color logic
            let color = 'bg-slate-400';
            if (label === 'YOU') {
               // Compare with truth at this index
               const correctVal = result.stress.truth[i];
               if (val === correctVal) {
                 color = isPrimary ? 'bg-emerald-500' : 'bg-slate-400';
               } else {
                 // Wrong stress placement
                 color = 'bg-rose-500';
               }
            } else {
               // Target (Truth) colors
               color = isPrimary ? 'bg-indigo-500' : 'bg-slate-400';
            }

            return (
              <div key={i} className="flex flex-col items-center justify-end h-full">
                <div className={`rounded-sm shadow-sm ${height} ${width} ${color} ${opacity} transition-all duration-500`}></div>
                <span className="text-[9px] text-slate-300 mt-1">{i + 1}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // Generate dynamic feedback
  const getFeedback = () => {
    if (result.isPhonemesCorrect && result.isStressCorrect) return "Excellent! Your pronunciation and stress are spot on.";
    if (!result.isStressCorrect) return "Watch your stress! You emphasized the wrong syllable. Compare the bars above.";
    if (!result.isPhonemesCorrect) return "Your stress is okay, but some sounds were unclear. Check the red phonemes.";
    return "Keep practicing! Try to mimic the reference audio's rhythm and sounds.";
  };

  return (
    <div className="w-full max-w-3xl bg-white rounded-2xl shadow-lg border border-slate-100 overflow-hidden mt-8 animate-fade-in-up">

      {/* Header Status */}
      <div className={`px-6 py-4 border-b flex justify-between items-center ${result.overallScore > 80 ? 'bg-emerald-50 border-emerald-100' : 'bg-slate-50 border-slate-100'}`}>
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-full ${result.overallScore > 70 ? 'bg-emerald-200' : 'bg-slate-200'}`}>
            <Activity className={`w-5 h-5 ${result.overallScore > 70 ? 'text-emerald-700' : 'text-slate-500'}`} />
          </div>
          <div>
            <h3 className="font-bold text-slate-800">Assessment Report</h3>
            <p className="text-xs text-slate-500 font-medium">Overall Score: <span className="text-indigo-600">{result.overallScore}/100</span></p>
          </div>
        </div>

        <span className={`px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wide border ${
          result.isPhonemesCorrect && result.isStressCorrect
            ? 'bg-emerald-100 text-emerald-700 border-emerald-200'
            : 'bg-rose-50 text-rose-600 border-rose-100'
        }`}>
          {result.isPhonemesCorrect && result.isStressCorrect ? 'Passed' : 'Needs Work'}
        </span>
      </div>

      <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-8">

        {/* Section 1: Phoneme Analysis */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-xs font-black text-slate-400 uppercase tracking-wider">Phoneme Breakdown</h4>
            {result.isPhonemesCorrect ? <CheckCircle className="w-5 h-5 text-emerald-500" /> : <XCircle className="w-5 h-5 text-rose-400" />}
          </div>

          <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
             {renderPhonemes()}
          </div>

          <p className="text-xs text-slate-400 mt-3 leading-relaxed">
            Scores near <span className="font-mono text-emerald-600">0.0</span> are perfect. <br/>
            Scores below <span className="font-mono text-rose-500">-2.0</span> indicate mispronunciation.
          </p>
        </div>

        {/* Section 2: Stress Analysis */}
        <div className="md:border-l md:border-slate-100 md:pl-8">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-xs font-black text-slate-400 uppercase tracking-wider">Syllable Stress</h4>
            {result.isStressCorrect ? <CheckCircle className="w-5 h-5 text-emerald-500" /> : <XCircle className="w-5 h-5 text-rose-400" />}
          </div>

          <div className="flex justify-center space-x-6 mb-6">
            {renderStressBars(result.stress.truth, 'TARGET')}
            {renderStressBars(result.stress.infer, 'YOU')}
          </div>

          <div className="bg-indigo-50 p-4 rounded-xl border border-indigo-100 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-1 h-full bg-indigo-400"></div>
            <p className="text-sm text-indigo-900 font-medium leading-relaxed relative z-10">
              <span className="block text-[10px] font-bold text-indigo-400 uppercase mb-1">Feedback</span>
              {getFeedback()}
            </p>
          </div>
        </div>

      </div>
    </div>
  );
};

export default ResultTable;
