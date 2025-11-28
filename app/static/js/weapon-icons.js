/* static/js/weapon-icons.js */
(function() {
    window.WEAPON_ICONS_LIB = {
        // AR: AKM (Refined Trigger & Guard)
        ar: `
            <!-- Receiver & Stock -->
            <path d="M5 40 L5 32 L15 32 L20 34 L20 30 L50 29 L50 26 L85 26 L85 25 L88 25 L88 29 L85 29 L85 31 L60 31 L60 34 L52 34 L52 36 L48 36 Q 45 42, 42 36 L 40 36 L42 46 L32 46 L35 36 L25 36 L20 34 L15 40 Z" />
            <!-- Trigger Blade -->
            <path d="M46 36 L46 39" stroke-width="1.5" />
            <!-- Gas Tube & Barrel -->
            <line x1="50" y1="28" x2="85" y2="28" />
            <line x1="85" y1="29.5" x2="95" y2="29.5" />
            <line x1="92" y1="29.5" x2="92" y2="25" />
            <!-- Curved Magazine -->
            <path d="M52 36 Q 55 50, 68 46 L 70 34" fill="none" stroke-width="2" />
        `,

        // SR: M24 (Long, Slender)
        sr: `
            <path d="M2 34 L2 28 L5 28 L5 30 L15 30 L20 32 L35 32 L35 29 L70 29 L70 32 L60 33 L55 33 L50 36 L40 36 L38 42 L32 42 L35 36 L25 36 L20 32 L5 34 Z" />
            <rect x="70" y="30" width="28" height="2" />
            <rect x="96" y="29" width="2" height="4" />
            <path d="M42 25 L40 21 L65 21 L63 25" />
            <line x1="42" y1="23" x2="63" y2="23" />
            <line x1="40" y1="29" x2="43" y2="26" stroke-linecap="round" />
        `,

        // SMG: Kriss Vector (Added Shoulder Stock)
        smg: `
            <!-- Shoulder Stock (Added) -->
            <path d="M25 23 L8 23 L8 34 L12 34 L12 27 L25 27" fill="none" stroke-width="1.5" />
            <!-- Main Receiver Body -->
            <path d="M25 30 L25 22 L75 22 L75 25 L82 25 L82 28 L75 28 L75 32 L65 32 L65 42 L55 42 L55 32 L45 32 L42 45 L35 45 L38 32 L25 32 Z" />
            <!-- Magazine (Glock style) -->
            <rect x="57" y="42" width="6" height="8" />
            <!-- Top Rail -->
            <line x1="25" y1="22" x2="75" y2="22" stroke-dasharray="2,2" />
            <!-- Ejection Port / Detail -->
            <rect x="45" y="26" width="10" height="3" />
        `,

        // SG: S1897 Pump Action
        sg: `
            <path d="M5 38 L5 30 L15 30 L20 33 L40 33 L40 30 L85 30 L85 36 L80 36 L80 38 L50 38 L45 42 L35 42 L40 36 L20 36 L15 38 Z" />
            <rect x="55" y="35" width="20" height="5" rx="2" />
            <line x1="40" y1="31" x2="90" y2="31" />
            <line x1="85" y1="35" x2="90" y2="35" />
        `,

        // LMG: M249 Realistic
        lmg: `
            <path d="M10 38 L15 30 L20 30 L20 34 L25 34 L25 30 L30 30 L30 40 L20 40 L15 38 Z" />
            <path d="M30 32 L85 32 L85 40 L55 40 L55 42 L35 42 L40 36 L30 36 Z" />
            <path d="M55 32 Q 62 28, 70 32 Q 77 28, 85 32" fill="none" />
            <line x1="85" y1="34" x2="95" y2="34" />
            <path d="M45 40 Q 42 50, 50 52 L 55 40" fill="none" stroke-width="2" />
            <line x1="80" y1="40" x2="78" y2="48" />
            <line x1="80" y1="40" x2="86" y2="48" />
        `,

        // HG: P1911
        hg: `
            <path d="M30 32 L30 24 L75 24 L75 32 L72 35 L30 35 Z" />
            <path d="M30 35 L25 48 L40 48 L44 35" />
            <path d="M44 35 Q 48 42, 54 35" fill="none" />
            <path d="M28 28 L30 28 L30 24" fill="none" />
        `,

        // None: Waveform
        none: '<path d="M10 30 L20 20 L30 40 L40 10 L50 50 L60 20 L70 40 L80 30 L90 30" style="fill:none; stroke-dasharray: 2,2;" />'
    };
})();