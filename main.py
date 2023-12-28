import time

import streamlit as st
import pandas as pd
import numpy as np
from predict import load_model, predict
from model import *


def main():
    model = load_model()
    model.eval()

    ## 타이틀 정의
    st.title("Sentence Similarity Prediction App")
    st.write("네이버 부스트캠프 AI Tech 6기 8주차 과제")

    ## 폼을 통해 두 문장을 입력 받아 문장 유사성 예측
    form = st.form('text_input')
    first_sentence = form.text_input('첫번째 문장을 입력하세요.')
    second_sentence = form.text_input('두번째 문장을 입력하세요.')
    submit = form.form_submit_button('결과 확인')
    if submit:
        # 잘못된 문장 처리
        if first_sentence == '' or second_sentence == '':
            st.toast('문장을 입력해주세요.')
            return

        # 예측 결과 출력
        try:
            pred = predict(model, first_sentence, second_sentence)
            st.subheader(f'예측 결과: {pred}')
            alert = st.success('유사도 예측 완료!')
            time.sleep(1)
            alert.empty()
        except:
            alert = st.error('문장 유사도 예측에 실패하였습니다.')
            time.sleep(1)
            alert.empty()


if __name__ == "__main__":
    main()