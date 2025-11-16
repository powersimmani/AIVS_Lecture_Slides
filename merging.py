import os
from pathlib import Path
from PyPDF2 import PdfMerger
import re

def merge_pdfs_by_lecture(input_dir, output_dir):
    """
    강의별로 PDF 파일을 순서대로 병합
    
    Args:
        input_dir: PDF 파일들이 있는 디렉토리 경로
        output_dir: 병합된 PDF를 저장할 디렉토리 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모든 PDF 파일 가져오기
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    
    # 강의별로 파일 그룹화
    lectures = {}
    for filename in pdf_files:
        # 강의 번호 추출 (예: Lecture01, Lecture02, ...)
        match = re.match(r'(Lecture\d{2})_(\d{2})_(.+)\.pdf', filename)
        if match:
            lecture_num = match.group(1)
            slide_num = int(match.group(2))
            
            if lecture_num not in lectures:
                lectures[lecture_num] = []
            
            lectures[lecture_num].append((slide_num, filename))
    
    # 각 강의별로 처리
    for lecture_num in sorted(lectures.keys()):
        print(f"\n처리 중: {lecture_num}")
        
        # 슬라이드 번호순으로 정렬
        slides = sorted(lectures[lecture_num], key=lambda x: x[0])
        
        # PDF 병합
        merger = PdfMerger()
        
        for slide_num, filename in slides:
            file_path = os.path.join(input_dir, filename)
            try:
                merger.append(file_path)
                print(f"  추가됨: {filename}")
            except Exception as e:
                print(f"  오류 발생 ({filename}): {e}")
        
        # 병합된 PDF 저장
        output_filename = f"{lecture_num}_merged.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            merger.write(output_path)
            merger.close()
            print(f"✓ 저장 완료: {output_filename} ({len(slides)}개 슬라이드)")
        except Exception as e:
            print(f"✗ 저장 실패 ({output_filename}): {e}")

# 사용 예시
if __name__ == "__main__":
    # 입력 및 출력 디렉토리 설정
    input_directory = "./pdf"  # PDF 파일들이 있는 폴더
    output_directory = "./pdf merged"  # 병합된 파일을 저장할 폴더
    
    print("=" * 60)
    print("PDF 병합 프로그램 시작")
    print("=" * 60)
    
    merge_pdfs_by_lecture(input_directory, output_directory)
    
    print("\n" + "=" * 60)
    print("모든 강의 병합 완료!")
    print("=" * 60)