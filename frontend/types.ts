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

export interface AssessmentResult {
  word: string;
  overallScore: number;
  isPhonemesCorrect: boolean;
  isStressCorrect: boolean;
  phonemesScore: Record<string, number>; // e.g. { "AH": -0.2, "T": -0.1 }
  stress: {
    truth: number[]; // [0, 1, 0]
    infer: number[]; // [0, 1, 0]
    syllableCount: number;
  };
}

export type AppState = 'MENU' | 'PRACTICE' | 'LOADING';
