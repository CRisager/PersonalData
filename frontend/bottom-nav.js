(function () {
	const NAV_HTML = `
		<nav class="bottom-nav" aria-label="Primary navigation">
			<svg class="icon profile" viewBox="0 0 24 24" aria-hidden="true">
				<circle cx="12" cy="8" r="3.7" />
				<path d="M4 19c1.9-2.8 4.8-4.2 8-4.2s6.1 1.4 8 4.2" />
				<circle cx="12" cy="12" r="10" />
			</svg>

			<svg class="record-icon" viewBox="0 0 36 36" aria-hidden="true">
				<circle cx="18" cy="18" r="15.2" fill="#ffffff" stroke="#27636d" stroke-width="3" />
				<circle cx="18" cy="18" r="9.2" fill="#27636d" stroke="none" />
			</svg>

			<svg class="icon home" viewBox="0 0 24 24" aria-hidden="true">
				<path d="M3.2 10.2L12 3.3l8.8 6.9V21H3.2z" />
			</svg>

			<svg class="icon search" viewBox="0 0 24 24" aria-hidden="true">
				<circle cx="10.5" cy="10.5" r="7" />
				<line x1="16" y1="16" x2="21" y2="21" />
				<line x1="7.8" y1="11" x2="7.8" y2="13.5" />
				<line x1="10.5" y1="8" x2="10.5" y2="13.5" />
				<line x1="13.2" y1="9.8" x2="13.2" y2="13.5" />
			</svg>

			<svg class="icon folder" viewBox="0 0 24 24" aria-hidden="true">
				<path d="M3 8h6.5l1.7 2H21v8.3A1.7 1.7 0 0 1 19.3 20H4.7A1.7 1.7 0 0 1 3 18.3z" />
			</svg>
		</nav>
	`;

	function injectNav() {
		const phone = document.querySelector(".phone");
		if (!phone) {
			return null;
		}

		const template = document.createElement("template");
		template.innerHTML = NAV_HTML.trim();
		const nav = template.content.firstElementChild;

		const overlay = phone.querySelector(".learn-info-overlay");
		if (overlay) {
			phone.insertBefore(nav, overlay);
		} else {
			phone.appendChild(nav);
		}

		return nav;
	}

	function applyActiveState(nav, activeNav) {
		if (!activeNav) {
			return;
		}

		const selectors = {
			profile: ".icon.profile",
			record: ".record-icon",
			home: ".icon.home",
			search: ".icon.search",
			folder: ".icon.folder",
		};

		const el = nav.querySelector(selectors[activeNav]);
		if (el) {
			el.classList.add("nav-active");
		}
	}

	function setupNavHandlers(nav) {
		const pathname = window.location.pathname.toLowerCase();
		const isInFrontend = pathname.includes("/frontend/");

		const indexHref = isInFrontend ? "../index.html" : "index.html";
		const insightsHref = isInFrontend ? "insights.html" : "frontend/insights.html";
		const insightsV2Href = isInFrontend ? "insights_v2.html" : "frontend/insights_v2.html";

		const isOnInsightsV2 = pathname.includes("insights_v2");
		const isOnPlot = pathname.includes("plot.html");
		const isOnInsights = pathname.includes("insights") && !isOnInsightsV2;
		const isOnProfile = pathname.includes("profile.html");
		const isOnHome = pathname.includes("home.html");

		const profileHref = isInFrontend ? "profile.html" : "frontend/profile.html";
		const homeHref = isInFrontend ? "home.html" : "frontend/home.html";

		const recordIcon = nav.querySelector(".record-icon");
		const profileIcon = nav.querySelector(".icon.profile");
		const homeIcon = nav.querySelector(".icon.home");
		const searchIcon = nav.querySelector(".icon.search");
		const folderIcon = nav.querySelector(".icon.folder");

		if (profileIcon) {
			profileIcon.style.cursor = "pointer";
			profileIcon.addEventListener("click", function (e) {
				e.stopPropagation();
				if (!isOnProfile) {
					window.location.href = profileHref;
				}
			});
		}

		if (homeIcon) {
			homeIcon.style.cursor = "pointer";
			homeIcon.addEventListener("click", function (e) {
				e.stopPropagation();
				if (!isOnHome) {
					window.location.href = homeHref;
				}
			});
		}

		if (recordIcon) {
			recordIcon.style.cursor = "pointer";
			recordIcon.addEventListener("click", function (e) {
				e.stopPropagation();
				if (typeof window.__showRecordingView === "function") {
					window.__showRecordingView();
				} else {
					window.location.href = indexHref;
				}
			});
		}

		if (searchIcon) {
			searchIcon.style.cursor = "pointer";
			searchIcon.addEventListener("click", function (e) {
				e.stopPropagation();
				if (isOnPlot) {
					window.location.href = "insights.html";
				} else if (isOnInsights) {
					window.location.reload();
				} else {
					window.location.href = insightsHref;
				}
			});
		}

		if (folderIcon) {
			folderIcon.style.cursor = "pointer";
			folderIcon.addEventListener("click", function (e) {
				e.stopPropagation();
				if (isOnInsightsV2) {
					window.location.href = "insights.html";
				} else {
					window.location.href = insightsV2Href;
				}
			});
		}
	}

	function fitPhoneToViewport() {
		const phoneStage = document.getElementById("phoneStage");
		if (!phoneStage) {
			return;
		}
		const BASE_PHONE_WIDTH = 404;
		const BASE_PHONE_HEIGHT = 873.7301635742188;
		const VIEWPORT_PAD = 16;
		const availableWidth = Math.max(window.innerWidth - VIEWPORT_PAD * 2, 1);
		const availableHeight = Math.max(window.innerHeight - VIEWPORT_PAD * 2, 1);
		const scale = Math.min(1, availableWidth / BASE_PHONE_WIDTH, availableHeight / BASE_PHONE_HEIGHT);
		phoneStage.style.setProperty("--phone-scale", String(scale));
	}

	function init() {
		fitPhoneToViewport();
		window.addEventListener("resize", fitPhoneToViewport);

		const nav = injectNav();
		if (!nav) {
			return;
		}

		const activeNav = document.body.dataset.activeNav || "";
		applyActiveState(nav, activeNav);
		setupNavHandlers(nav);
	}

	if (document.readyState === "loading") {
		document.addEventListener("DOMContentLoaded", init);
	} else {
		init();
	}
})();
