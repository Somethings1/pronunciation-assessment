import { DifficultyBand, WordData } from '../types';

// A small subset of words for demonstration purposes.
// In a real app, this would be a database or large JSON file.
export const WORD_DATABASE: WordData[] = [
  // A1
  { id: '1', word: 'Apple', band: DifficultyBand.A1, ipa: '/ˈæp.əl/', phonemes: 'AE1 P AH0 L', syllables: ['Ap', 'ple'], stressPattern: '10' },
  { id: '2', word: 'Hello', band: DifficultyBand.A1, ipa: '/həˈloʊ/', phonemes: 'HH AH0 L OW1', syllables: ['Hel', 'lo'], stressPattern: '01' },
  { id: '3', word: 'Water', band: DifficultyBand.A1, ipa: '/ˈwɔː.tər/', phonemes: 'W AO1 T ER0', syllables: ['Wa', 'ter'], stressPattern: '10' },
  
  // A2
  { id: '4', word: 'Decide', band: DifficultyBand.A2, ipa: '/dɪˈsaɪd/', phonemes: 'D IH2 S AY1 D', syllables: ['De', 'cide'], stressPattern: '01' },
  { id: '5', word: 'Garden', band: DifficultyBand.A2, ipa: '/ˈɡɑːr.dən/', phonemes: 'G AA1 R D AH0 N', syllables: ['Gar', 'den'], stressPattern: '10' },
  
  // B1
  { id: '6', word: 'Delicious', band: DifficultyBand.B1, ipa: '/dɪˈlɪʃ.əs/', phonemes: 'D IH0 L IH1 SH AH0 S', syllables: ['De', 'li', 'cious'], stressPattern: '010' },
  { id: '7', word: 'Adventure', band: DifficultyBand.B1, ipa: '/ədˈven.tʃər/', phonemes: 'AH0 D V EH1 N CH ER0', syllables: ['Ad', 'ven', 'ture'], stressPattern: '010' },

  // B2
  { id: '8', word: 'Significant', band: DifficultyBand.B2, ipa: '/sɪɡˈnɪf.ɪ.kənt/', phonemes: 'S IH0 G N IH1 F IH0 K AH0 N T', syllables: ['Sig', 'ni', 'fi', 'cant'], stressPattern: '0100' },
  { id: '9', word: 'Capacity', band: DifficultyBand.B2, ipa: '/kəˈpæs.ə.t̬i/', phonemes: 'K AH0 P AE1 S AH0 T IY0', syllables: ['Ca', 'pa', 'ci', 'ty'], stressPattern: '0100' },

  // C1
  { id: '10', word: 'Inevitably', band: DifficultyBand.C1, ipa: '/ɪˈnev.ə.t̬ə.bli/', phonemes: 'IH2 N EH1 V AH0 T AH0 B L IY0', syllables: ['In', 'ev', 'i', 'ta', 'bly'], stressPattern: '21000' },
  { id: '11', word: 'Hypothesis', band: DifficultyBand.C1, ipa: '/haɪˈpɑː.θə.sɪs/', phonemes: 'H AY0 P AA1 TH AH0 S AH0 S', syllables: ['Hy', 'po', 'the', 'sis'], stressPattern: '0100' },

  // C2
  { id: '12', word: 'Procrastination', band: DifficultyBand.C2, ipa: '/proʊˌkræs.təˈneɪ.ʃən/', phonemes: 'P R OW0 K R AE2 S T AH0 N EY1 SH AH0 N', syllables: ['Pro', 'cras', 'ti', 'na', 'tion'], stressPattern: '020010' },
  { id: '13', word: 'Entrepreneurial', band: DifficultyBand.C2, ipa: '/ˌɑːn.trə.prəˈnʊr.i.əl/', phonemes: 'AA2 N T R AH0 P R AH0 N ER1 IY0 AH0 L', syllables: ['En', 'tre', 'pre', 'neur', 'i', 'al'], stressPattern: '200100' },
];

export const getWordsByBand = (band: DifficultyBand): WordData[] => {
  return WORD_DATABASE.filter((w) => w.band === band);
};

export const getRandomWord = (band: DifficultyBand, excludeIds: string[] = []): WordData | null => {
  const words = getWordsByBand(band).filter(w => !excludeIds.includes(w.id));
  if (words.length === 0) return null;
  return words[Math.floor(Math.random() * words.length)];
};