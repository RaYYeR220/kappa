export function Mark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 22 22" aria-hidden="true" focusable="false">
      <g fill="currentColor">
        <circle cx="4" cy="4" r="1.4" />
        <circle cx="11" cy="4" r="1.4" />
        <circle cx="18" cy="4" r="1.4" />
        <circle cx="4" cy="11" r="1.4" />
        <circle cx="18" cy="11" r="1.4" />
        <circle cx="4" cy="18" r="1.4" />
        <circle cx="11" cy="18" r="1.4" />
        <circle cx="18" cy="18" r="1.4" />
      </g>
      <ellipse
        cx="12.4"
        cy="11.6"
        rx="3.6"
        ry="2.6"
        fill="#f9786a"
        stroke="#bc4a3f"
        strokeWidth="1.2"
        transform="rotate(-24 12.4 11.6)"
      />
    </svg>
  );
}
