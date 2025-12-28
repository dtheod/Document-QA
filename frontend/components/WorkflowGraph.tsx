
import React from 'react';
import { WorkflowStep } from '../types';

interface WorkflowGraphProps {
  currentStep: WorkflowStep;
}

const WorkflowGraph: React.FC<WorkflowGraphProps> = ({ currentStep }) => {
  const nodes = [
    { id: WorkflowStep.UPLOAD, label: 'Upload' },
    { id: WorkflowStep.EXTRACT, label: 'Extract' },
    { id: WorkflowStep.ANALYZE, label: 'Analyze' },
    { id: WorkflowStep.INDEX, label: 'Index' },
    { id: WorkflowStep.READY, label: 'Ready' }
  ];

  const getStepIndex = (step: WorkflowStep) => {
    const idx = nodes.findIndex(n => n.id === step);
    if (idx === -1) {
       if (step === WorkflowStep.RETRIEVING) return 3.5;
       if (step === WorkflowStep.GENERATING) return 4;
       return -1;
    }
    return idx;
  };

  const currentIndex = getStepIndex(currentStep);

  return (
    <div className="flex items-center justify-between w-full max-w-2xl mx-auto px-4 py-8">
      {nodes.map((node, index) => (
        <React.Fragment key={node.id}>
          <div className="flex flex-col items-center relative z-10">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-500 
              ${index <= currentIndex 
                ? 'bg-blue-600 border-blue-600 text-white' 
                : 'bg-white border-gray-300 text-gray-400'
              }
              ${index === currentIndex ? 'ring-4 ring-blue-100 animate-pulse' : ''}
            `}>
              {index < currentIndex ? (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <span className="text-sm font-bold">{index + 1}</span>
              )}
            </div>
            <span className={`mt-2 text-xs font-medium uppercase tracking-wider ${index <= currentIndex ? 'text-blue-600' : 'text-gray-400'}`}>
              {node.label}
            </span>
          </div>
          {index < nodes.length - 1 && (
            <div className="flex-1 h-0.5 bg-gray-200 mx-2 relative -mt-6">
              <div 
                className="absolute top-0 left-0 h-full bg-blue-600 transition-all duration-500" 
                style={{ width: index < currentIndex ? '100%' : '0%' }}
              />
            </div>
          )}
        </React.Fragment>
      ))}
    </div>
  );
};

export default WorkflowGraph;
