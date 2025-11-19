export enum DifficultyBand {
  A1 = 'A1',
  A2 = 'A2',
  B1 = 'B1',
  B2 = 'B2',
  C1 = 'C1',
  C2 = 'C2',
}

export interface WordData {
  id: string;
  word: string;
  band: DifficultyBand;
  ipa: string;
  phonemes: string; // CMU Arpabet format space separated
  syllables: string[]; // Array of syllables for stress vis
  stressPattern: string; // e.g. "010" (0=unstressed, 1=primary, 2=secondary)
}

export interface PhonemeError {
  index: number; // Index in the target phoneme sequence
  type: 'substitution' | 'deletion' | 'insertion' | 'correct';
  expected: string;
  actual: string;
}

export interface AssessmentResult {
  isStressCorrect: boolean;
  isPhonemesCorrect: boolean;
  detectedPhonemes: string[];
  detectedStressPattern: string;
  phonemeErrors: PhonemeError[];
  feedback: string;
}

export type AppState = 'MENU' | 'PRACTICE' | 'LOADING';
