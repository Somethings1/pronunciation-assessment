import React from 'react';
import { CheckCircle, XCircle, Activity, AlertCircle } from './Icons';
import { AssessmentResult, WordData, PhonemeError } from '../types';

interface ResultTableProps {
  result: AssessmentResult;
  targetWord: WordData;
}

const ResultTable: React.FC<ResultTableProps> = ({ result, targetWord }) => {
  
  // Helper to render Phoneme with status color
  const renderPhoneme = (p: string, index: number, errors: PhonemeError[]) => {
    const error = errors.find(e => e.index === index);
    let colorClass = "bg-green-100 text-green-800 border-green-200"; // Default correct
    
    if (error) {
      if (error.type === 'substitution') {
         // Yellow is fine for visualization, but it's an error
         colorClass = "bg-yellow-100 text-yellow-800 border-yellow-200 ring-1 ring-yellow-300";
      }
      if (error.type === 'deletion') colorClass = "bg-red-100 text-red-800 border-red-200 line-through decoration-red-500 font-bold";
      if (error.type === 'insertion') colorClass = "bg-orange-100 text-orange-800 border-orange-200 font-bold";
    }

    return (
      <span key={`${p}-${index}`} className={`inline-block px-2 py-1 rounded border text-sm font-mono mr-1 mb-1 transition-colors ${colorClass}`}>
        {p}
      </span>
    );
  };

  const renderStressPattern = (pattern: string, isTarget: boolean) => {
    return pattern.split('').map((char, i) => {
      const isPrimary = char === '1';
      const isSecondary = char === '2';
      
      // Visual logic: Highlight Primary ('1'). Fade others.
      const height = isPrimary ? 'h-10 w-4' : isSecondary ? 'h-6 w-3' : 'h-3 w-3';
      const opacity = isPrimary ? 'opacity-100' : 'opacity-40';
      
      let color = 'bg-slate-300';
      if (!isTarget) {
         // Detected pattern colors
         if (result.isStressCorrect) {
             color = isPrimary ? 'bg-emerald-500' : 'bg-emerald-300';
         } else {
             // If stress failed, highlight the wrong primary in red
             color = isPrimary ? 'bg-rose-500' : 'bg-slate-300';
         }
      } else {
        // Target pattern colors
        color = isPrimary ? 'bg-indigo-500' : 'bg-slate-300';
      }
      
      return (
        <div key={i} className="flex flex-col items-center mx-1 justify-end h-16">
           <div className={`rounded-t-sm shadow-sm ${height} ${color} ${opacity} transition-all duration-500 ease-out`}></div>
           <span className={`text-[10px] mt-2 font-mono ${isPrimary ? 'text-slate-800 font-bold' : 'text-slate-400'}`}>
             {i+1}
           </span>
        </div>
      );
    });
  };

  return (
    <div className="w-full max-w-2xl bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden mt-6 animate-fade-in">
      {/* Header */}
      <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex justify-between items-center">
        <h3 className="font-semibold text-slate-800 flex items-center gap-2">
           <Activity className="w-4 h-4 text-indigo-500" />
           Assessment Result
        </h3>
        <span className={`text-sm font-bold px-4 py-1.5 rounded-full shadow-sm border ${result.isPhonemesCorrect && result.isStressCorrect ? 'bg-emerald-100 text-emerald-700 border-emerald-200' : 'bg-rose-100 text-rose-700 border-rose-200'}`}>
          {result.isPhonemesCorrect && result.isStressCorrect ? 'Passed' : 'Try Again'}
        </span>
      </div>

      <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-8 relative">
        
        {/* Column 1: Phoneme Level */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-xs font-extrabold text-slate-400 uppercase tracking-wider">Phoneme Accuracy</h4>
            <div className="flex items-center">
                {result.isPhonemesCorrect ? (
                   <CheckCircle className="w-5 h-5 text-emerald-500" />
                ) : (
                   <XCircle className="w-5 h-5 text-rose-500" />
                )}
            </div>
          </div>
          
          {/* Comparison View */}
          <div className="space-y-4">
            <div>
                <div className="text-[10px] font-bold text-slate-400 mb-1">TARGET</div>
                <div className="flex flex-wrap">
                {targetWord.phonemes.split(' ').map((p, i) => (
                    <span key={i} className="inline-block px-2 py-1 rounded border border-slate-100 bg-slate-50 text-slate-400 text-sm font-mono font-bold mr-1 mb-1">
                    {p}
                    </span>
                ))}
                </div>
            </div>

            <div>
                <div className="text-[10px] font-bold text-slate-400 mb-1">YOUR ATTEMPT</div>
                <div className="flex flex-wrap">
                    {targetWord.phonemes.split(' ').map((p, i) => renderPhoneme(p, i, result.phonemeErrors))}
                    {/* Show Insertions */}
                    {result.phonemeErrors.filter(e => e.type === 'insertion').map((e, i) => (
                        <span key={`ins-${i}`} className="inline-block px-2 py-1 rounded border bg-orange-100 text-orange-800 border-orange-200 text-sm font-mono font-bold mr-1 mb-1">
                        +{e.actual}
                        </span>
                    ))}
                </div>
            </div>
          </div>
          
          <div className="mt-4 text-xs text-slate-400 space-y-1">
             <div className="flex items-center"><span className="inline-block w-2 h-2 bg-yellow-200 rounded-full mr-2"></span> Substitution</div>
             <div className="flex items-center"><span className="inline-block w-2 h-2 bg-red-200 rounded-full mr-2"></span> Deletion</div>
             <div className="flex items-center"><span className="inline-block w-2 h-2 bg-orange-200 rounded-full mr-2"></span> Insertion</div>
          </div>
        </div>

        {/* Column 2: Syllable Stress */}
        <div className="border-t md:border-t-0 md:border-l border-slate-100 pt-6 md:pt-0 md:pl-8">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-xs font-extrabold text-slate-400 uppercase tracking-wider">Primary Stress</h4>
            {result.isStressCorrect ? 
              <CheckCircle className="w-5 h-5 text-emerald-500" /> : 
              <XCircle className="w-5 h-5 text-rose-500" />
            }
          </div>

          <div className="flex justify-center space-x-8 mb-6 bg-slate-50 rounded-lg p-4 border border-slate-100">
             <div className="flex flex-col items-center">
                <span className="text-[10px] font-bold text-slate-400 mb-2">TARGET</span>
                <div className="flex items-end">
                   {renderStressPattern(targetWord.stressPattern, true)}
                </div>
             </div>
             
             <div className="h-full w-px bg-slate-200 mx-2"></div>

             <div className="flex flex-col items-center">
                <span className="text-[10px] font-bold text-slate-400 mb-2">YOU</span>
                <div className="flex items-end">
                   {renderStressPattern(result.detectedStressPattern, false)}
                </div>
             </div>
          </div>

           <div className="bg-indigo-50 p-3 rounded-lg border border-indigo-100">
              <p className="text-sm text-indigo-800 leading-relaxed">
                  <span className="font-bold block text-indigo-900 text-xs uppercase mb-1">Feedback</span>
                  {result.feedback}
              </p>
           </div>
        </div>

      </div>
    </div>
  );
};

export default ResultTable;