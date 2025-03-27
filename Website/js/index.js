window.addEventListener('scroll', function() {
    const header = document.querySelector('header');
    const scrollPosition = window.scrollY;
    
    if (scrollPosition > 100) {
        header.classList.add('scrolled-nav');
        header.classList.remove('bg-transparent');
    } else {
        header.classList.remove('scrolled-nav');
        header.classList.add('bg-transparent');
    }
});