#!/usr/bin/env python3
"""
Universal HTML to PDF Converter for Lecture Folders
각 LectureXX 폴더의 HTML 파일들을 PDF로 변환합니다.

사용법:
    python convert_lectures_to_pdf.py

이 스크립트는:
1. 현재 디렉토리의 모든 LectureXX 폴더를 찾습니다
2. 각 폴더의 LectureXX_XX.html 파일들을 PDF로 변환합니다
3. pdf/LectureXX_XX.pdf 형태로 저장합니다
4. slideshow, readme 등 불필요한 파일은 제외합니다
"""

import os
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

# 제외할 파일 패턴
EXCLUDE_PATTERNS = [
    'slideshow',
    'readme',
    'index',
]

def should_exclude(filename):
    """파일명이 제외 패턴에 해당하는지 확인"""
    filename_lower = filename.lower()
    for pattern in EXCLUDE_PATTERNS:
        if pattern in filename_lower:
            return True
    return False

def extract_lecture_number(folder_name):
    """폴더명에서 강의 번호 추출 (예: Lecture02 -> 02)"""
    match = re.search(r'Lecture(\d+)', folder_name, re.IGNORECASE)
    return match.group(1) if match else None

def convert_html_to_pdf(html_file_path, output_pdf_path):
    """
    Playwright를 사용하여 HTML을 PDF로 변환
    
    Args:
        html_file_path: 입력 HTML 파일 경로
        output_pdf_path: 출력 PDF 파일 경로
    
    Returns:
        bool: 성공 여부
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # HTML 파일 열기
            page.goto(f'file://{html_file_path}')
            
            # PDF로 저장 (A4 사이즈, 가로 모드)
            page.pdf(
                path=str(output_pdf_path),
                format='A4',
                landscape=True,
                print_background=True,
                margin={
                    'top': '0mm',
                    'right': '0mm',
                    'bottom': '0mm',
                    'left': '0mm'
                }
            )
            
            browser.close()
        
        return True
        
    except Exception as e:
        print(f"    ✗ 오류: {str(e)}")
        return False

def process_lecture_folder(lecture_folder, output_base_dir):
    """
    단일 Lecture 폴더 처리
    
    Args:
        lecture_folder: Lecture 폴더 경로
        output_base_dir: PDF 출력 기본 디렉토리
    
    Returns:
        tuple: (성공 개수, 실패 개수)
    """
    folder_name = lecture_folder.name
    lecture_num = extract_lecture_number(folder_name)
    
    if not lecture_num:
        print(f"⚠ 경고: {folder_name}에서 강의 번호를 찾을 수 없습니다.")
        return 0, 0
    
    # Lecture 패턴에 맞는 HTML 파일만 찾기 (예: Lecture02_01_*.html)
    pattern = f"Lecture{lecture_num}_*.html"
    html_files = sorted([
        f for f in lecture_folder.glob(pattern)
        if not should_exclude(f.name)
    ])
    
    if not html_files:
        print(f"  ℹ {folder_name}: 변환할 HTML 파일이 없습니다.")
        return 0, 0
    
    print(f"\n{'='*80}")
    print(f"📁 {folder_name} 폴더 처리 중... ({len(html_files)}개 파일)")
    print(f"{'='*80}")
    
    success_count = 0
    fail_count = 0
    
    for i, html_file in enumerate(html_files, 1):
        # PDF 파일명 생성: LectureXX_YY.pdf
        pdf_filename = html_file.stem + ".pdf"
        output_pdf = output_base_dir / pdf_filename
        
        print(f"  [{i:2d}/{len(html_files)}] {html_file.name:70s}", end=" ")
        
        if convert_html_to_pdf(html_file.absolute(), output_pdf.absolute()):
            file_size = output_pdf.stat().st_size / 1024  # KB
            print(f"✓ ({file_size:6.1f} KB)")
            success_count += 1
        else:
            print(f"✗")
            fail_count += 1
    
    return success_count, fail_count

def main():
    """메인 함수"""
    # 현재 작업 디렉토리
    current_dir = Path.cwd()
    
    # PDF 출력 디렉토리 생성
    output_dir = current_dir / "pdf"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("🎓 Lecture HTML to PDF Converter")
    print("="*80)
    print(f"📂 작업 디렉토리: {current_dir}")
    print(f"📄 PDF 저장 위치: {output_dir}")
    
    # LectureXX 패턴의 폴더 찾기
    lecture_folders = sorted([
        d for d in current_dir.iterdir()
        if d.is_dir() and re.match(r'Lecture\d+', d.name, re.IGNORECASE)
    ])
    
    if not lecture_folders:
        print("\n⚠ LectureXX 형식의 폴더를 찾을 수 없습니다.")
        print("  이 스크립트는 Lecture 폴더들이 있는 디렉토리에서 실행해야 합니다.")
        return
    
    print(f"\n발견된 Lecture 폴더: {len(lecture_folders)}개")
    for folder in lecture_folders:
        print(f"  - {folder.name}")
    
    # 전체 통계
    total_success = 0
    total_fail = 0
    
    # 각 Lecture 폴더 처리
    for lecture_folder in lecture_folders:
        success, fail = process_lecture_folder(lecture_folder, output_dir)
        total_success += success
        total_fail += fail
    
    # 최종 결과 출력
    print("\n" + "="*80)
    print("📊 최종 변환 결과")
    print("="*80)
    print(f"  ✅ 성공: {total_success}개")
    print(f"  ❌ 실패: {total_fail}개")
    print(f"  📁 총 처리된 파일: {total_success + total_fail}개")
    print(f"\n💾 모든 PDF 파일이 '{output_dir}' 폴더에 저장되었습니다.")
    
    # 생성된 PDF 파일 목록
    pdf_files = sorted(output_dir.glob("*.pdf"))
    if pdf_files:
        print(f"\n📄 생성된 PDF 파일 목록 ({len(pdf_files)}개):")
        
        # Lecture 번호별로 그룹화
        current_lecture = None
        for pdf_file in pdf_files:
            lecture_match = re.match(r'Lecture(\d+)_', pdf_file.name)
            lecture_num = lecture_match.group(1) if lecture_match else None
            
            if lecture_num != current_lecture:
                if current_lecture is not None:
                    print()
                current_lecture = lecture_num
                print(f"\n  [Lecture {lecture_num}]")
            
            file_size = pdf_file.stat().st_size / 1024
            print(f"    • {pdf_file.name} ({file_size:.1f} KB)")
    
    print("\n" + "="*80)
    print("✅ 모든 작업이 완료되었습니다!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()