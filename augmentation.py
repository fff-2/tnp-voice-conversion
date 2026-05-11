import math

import librosa
import parselmouth
import soundfile as sf
from parselmouth.praat import call


def augment_pitch_and_formant(audio_path, output_path, formant_ratio, pitch_ratio):
    """
    formant_ratio: 성도 길이 조절 (예: 0.85 ~ 1.15)
    pitch_ratio: 피치 조절 (예: 0.8 ~ 1.2 -> 1.2는 피치를 20% 높임)
    """
    # 1. 오디오 강제 모노 로드
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    sound = parselmouth.Sound(y, sr)

    # 2. 원본 오디오의 평균 피치(Median Pitch) 추출
    pitch_obj = sound.to_pitch()
    median_pitch = call(pitch_obj, "Get quantile", 0.0, 0.0, 0.5, "Hertz")

    # 3. 방어 로직: 오디오가 너무 짧거나 무음만 있어서 피치가 안 잡히는 경우(NaN)
    if math.isnan(median_pitch):
        target_pitch = 0.0  # 피치 변환을 건너뜀
    else:
        target_pitch = median_pitch * pitch_ratio  # 예: 150Hz * 1.2배 = 180Hz

    # 4. Praat Change Gender 호출!
    # 파라미터: f0min, f0max, formant_ratio, target_pitch, pitch_range_ratio, duration_factor
    augmented_sound = call(
        sound, "Change gender", 75.0, 600.0, formant_ratio, target_pitch, 1.0, 1.0
    )

    # 5. 정수형 샘플링 레이트로 안전하게 저장
    sf.write(
        output_path, augmented_sound.values[0], int(augmented_sound.sampling_frequency)
    )


# 실행 테스트
if __name__ == "__main__":
    # 포먼트를 1.1배(가볍게), 피치도 1.2배(높게) 올려서 타겟 화자 보호막 만들기
    augment_pitch_and_formant(
        audio_path="datasets/wav48_silence_trimmed/p224/1_0000.wav",
        output_path="augmented_pitch_formant.wav",
        formant_ratio=0.7,
        pitch_ratio=0.9,
    )
