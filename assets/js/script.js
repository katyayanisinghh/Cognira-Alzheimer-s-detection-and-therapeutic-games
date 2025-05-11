
const swiper = new Swiper('.swiper-container', {
  slidesPerView: 3, // Shows three full cards
  spaceBetween: 30, 
  centeredSlides: false, 
  loop: false, 
  navigation: {
    nextEl: '.next-btn', //next button
    prevEl: '.prev-btn', //previous button
  },
  pagination: {
    el: '.swiper-pagination',
    clickable: true,
  },

  //Breakpoints for responsive design

  breakpoints: {
    0: {
        slidesPerView: 1
    },
    500: {
      slidesPerView: 1.5
    },

    768: {
        slidesPerView: 2.2
    },
    1200: {
        slidesPerView: 3.2
    },
}
});

// script.js
document.addEventListener("DOMContentLoaded", () => {
  const navbarContainer = document.getElementById("navbar-container");

  fetch("./nabar.html")
      .then((response) => response.text())
      .then((html) => {
          navbarContainer.innerHTML = html;

          // Attach any additional event listeners after loading the navbar
          const toggleBtn = document.querySelector(".toggle-btn");
          const dropDownMenu = document.querySelector(".dropdown-menu");

          toggleBtn.onclick = function () {
              if (dropDownMenu.classList.contains("open")) {
                  dropDownMenu.classList.remove("open");
              } else {
                  dropDownMenu.classList.add("open");
              }
          };
      })
      .catch((error) => console.error("Error loading navbar:", error));
});

