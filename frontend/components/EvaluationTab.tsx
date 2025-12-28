
import React from 'react';
import { EvaluationResult } from '../types';

interface EvaluationTabProps {
  evaluations: EvaluationResult[];
  isLoading: boolean;
  onRunEvaluation: () => void;
}

const EvaluationTab: React.FC<EvaluationTabProps> = ({ evaluations, isLoading, onRunEvaluation }) => {
  if (isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 space-y-4">
        <div className="relative">
          <div className="w-16 h-16 border-4 border-blue-100 border-t-blue-600 rounded-full animate-spin"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <svg className="w-6 h-6 text-blue-600 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          </div>
        </div>
        <div className="text-center">
          <p className="text-slate-800 font-bold text-lg">Running Audit</p>
          <p className="text-slate-500 text-sm max-w-xs mx-auto">Gemini is analyzing the document against predefined quality and content criteria...</p>
        </div>
      </div>
    );
  }

  if (evaluations.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-12 text-center max-w-2xl mx-auto">
        <div className="w-24 h-24 bg-white shadow-xl shadow-slate-200 border border-slate-100 rounded-[2.5rem] flex items-center justify-center mb-8">
          <svg className="w-12 h-12 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
        </div>
        <h3 className="text-2xl font-bold text-slate-900 mb-4">Automated Evaluation Suite</h3>
        <p className="text-slate-500 mb-8 leading-relaxed">
          Perform a structured audit of your document. This process will evaluate objective clarity,
          actionable insights, risk identification, and target audience alignment using advanced LLM reasoning.
        </p>
        <button
          onClick={onRunEvaluation}
          className="group relative px-8 py-4 bg-white border-2 border-blue-600 text-blue-600 rounded-2xl font-bold overflow-hidden transition-all hover:bg-blue-600 hover:text-white"
        >
          <span className="relative z-10 flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Start Document Audit
          </span>
        </button>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      <div className="grid grid-cols-1 gap-6 max-w-5xl mx-auto">
        <div className="flex justify-between items-end mb-2">
          <div>
            <h2 className="text-xl font-bold text-slate-800">Audit Results</h2>
            <p className="text-slate-500 text-sm">Automated analysis completed successfully.</p>
          </div>
          <button
            onClick={onRunEvaluation}
            className="text-xs font-bold text-blue-600 hover:text-blue-700 underline underline-offset-4"
          >
            Re-run Evaluation
          </button>
        </div>

        {evaluations.map((evalItem, index) => (
          <div key={index} className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm hover:shadow-md transition-shadow">
            <div className="px-6 py-4 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
              <h4 className="font-bold text-slate-800 flex items-center gap-2">
                <span className="bg-blue-100 text-blue-700 text-[10px] px-2 py-0.5 rounded-full uppercase tracking-tighter font-black">Audit {index + 1}</span>
                {evalItem.question}
              </h4>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Confidence</span>
                <div className={`text-sm font-bold px-3 py-1 rounded-lg ${evalItem.score > 80 ? 'bg-green-100 text-green-700' :
                  evalItem.score > 50 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'
                  }`}>
                  {evalItem.score}%
                </div>
              </div>
            </div>
            <div className="p-6 space-y-4">
              <div>
                <h5 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">Detailed Metrics</h5>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
                  {[
                    { label: 'Faithfulness', value: evalItem.faithfulness, color: 'blue' },
                    { label: 'Answer Correctness', value: evalItem.answer_correctness, color: 'green' },
                    { label: 'Relevancy', value: evalItem.answer_relevancy, color: 'indigo' },
                    { label: 'Context Precision', value: evalItem.context_precision, color: 'purple' },
                    { label: 'Context Recall', value: evalItem.context_recall, color: 'pink' }
                  ].map((metric) => (
                    metric.value !== undefined && (
                      <div key={metric.label} className={`bg-${metric.color}-50 border border-${metric.color}-100 rounded-xl p-3 text-center`}>
                        <div className={`text-2xl font-bold text-${metric.color}-600`}>
                          {(metric.value * 100).toFixed(0)}%
                        </div>
                        <div className={`text-[10px] uppercase font-bold text-${metric.color}-400 tracking-wider`}>
                          {metric.label}
                        </div>
                      </div>
                    )
                  ))}
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">Model Answer</h5>
                    <p className="text-slate-700 leading-relaxed text-sm bg-slate-50 p-4 rounded-xl border border-slate-100 h-full">
                      {evalItem.response}
                    </p>
                  </div>
                  {evalItem.ground_truth && (
                    <div>
                      <h5 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">Ground Truth</h5>
                      <p className="text-green-800 leading-relaxed text-sm bg-green-50/50 p-4 rounded-xl border border-green-100 h-full">
                        {evalItem.ground_truth}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EvaluationTab;
