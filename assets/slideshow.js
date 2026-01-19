/* Slideshow Common JavaScript */
/* slides 배열은 각 HTML 파일에서 정의해야 합니다 */

let currentSlideIndex = 0;

function initializeSlides() {
    const container = document.getElementById('slideContainer');

    // 모든 슬라이드를 미리 생성
    slides.forEach((slide, index) => {
        const slideDiv = document.createElement('div');
        slideDiv.className = 'slide';
        slideDiv.id = `slide-${index}`;

        const iframe = document.createElement('iframe');
        iframe.src = encodeURI(slide.file);
        iframe.loading = 'lazy';

        slideDiv.appendChild(iframe);
        container.appendChild(slideDiv);
    });
}

function showSlide(index) {
    // 모든 슬라이드 숨기기
    const allSlides = document.querySelectorAll('.slide');
    allSlides.forEach(slide => {
        slide.classList.remove('active');
    });

    // 현재 슬라이드 표시
    const currentSlide = document.getElementById(`slide-${index}`);
    if (currentSlide) {
        currentSlide.classList.add('active');
    }

    // 제목 업데이트 (파트 정보 포함)
    const slideInfo = slides[index];
    let titleText = slideInfo.title;
    if (slideInfo.part) {
        titleText = `${slideInfo.title} | Part ${slideInfo.part}`;
    }
    document.getElementById('slideTitle').textContent = titleText;

    // 카운터 업데이트
    document.getElementById('currentSlide').textContent = index + 1;
    document.getElementById('totalSlides').textContent = slides.length;

    // 진행 바 업데이트
    const progress = ((index + 1) / slides.length) * 100;
    document.getElementById('progressFill').style.width = progress + '%';

    // URL 해시 업데이트 (1-base index)
    window.location.hash = (index + 1).toString();

    // 버튼 상태 업데이트
    document.getElementById('prevBtn').disabled = index === 0;
    document.getElementById('nextBtn').disabled = index === slides.length - 1;

    // 메뉴 선택 상태 업데이트
    const listItems = document.querySelectorAll('.slide-list-item');
    listItems.forEach((item, i) => {
        if (i === index) {
            item.classList.add('current');
        } else {
            item.classList.remove('current');
        }
    });
}

function changeSlide(direction) {
    const newIndex = currentSlideIndex + direction;

    if (newIndex >= 0 && newIndex < slides.length) {
        currentSlideIndex = newIndex;
        showSlide(currentSlideIndex);
    }
}

// 특정 슬라이드로 이동
function goToSlide(index) {
    if (index >= 0 && index < slides.length) {
        currentSlideIndex = index;
        showSlide(currentSlideIndex);
    }
}

// 슬라이드 목록 생성
function buildSlideList() {
    const slideList = document.getElementById('slideList');
    slideList.innerHTML = '';
    slides.forEach((slide, index) => {
        const item = document.createElement('div');
        item.className = 'slide-list-item';
        item.textContent = `${index + 1}. ${slide.title}`;
        item.onclick = () => goToSlide(index);
        slideList.appendChild(item);
    });
}

// 메뉴 토글
function toggleMenu() {
    document.getElementById('slideList').classList.toggle('active');
}

// 전체화면 토글
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}

// 키보드 내비게이션
document.addEventListener('keydown', (event) => {
    switch(event.key) {
        case 'ArrowLeft':
        case 'ArrowUp':
            changeSlide(-1);
            break;
        case 'ArrowRight':
        case 'ArrowDown':
        case ' ':
            event.preventDefault();
            changeSlide(1);
            break;
        case 'Home':
            currentSlideIndex = 0;
            showSlide(currentSlideIndex);
            break;
        case 'End':
            currentSlideIndex = slides.length - 1;
            showSlide(currentSlideIndex);
            break;
    }
});

// ESC로 메뉴 닫기
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.getElementById('slideList').classList.remove('active');
    }
});

// 페이지 로드 시 슬라이드 초기화
window.addEventListener('load', () => {
    const hashIndex = parseInt(window.location.hash.replace('#',''), 10);
    if (!isNaN(hashIndex) && hashIndex >= 1 && hashIndex <= slides.length) {
        currentSlideIndex = hashIndex - 1;
    }
    initializeSlides();
    buildSlideList();
    showSlide(currentSlideIndex);
});

// 스페이스바로 스크롤 방지
window.addEventListener('keydown', (e) => {
    if(e.key === ' ' && e.target === document.body) {
        e.preventDefault();
    }
});
