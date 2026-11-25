import type { ReactNode } from "react";
export type IconName =
  | "hand"
  | "scan"
  | "book"
  | "home"
  | "info"
  | "sun"
  | "moon"
  | "arrow"
  | "camera"
  | "pause"
  | "play"
  | "mirror"
  | "chevron"
  | "settings"
  | "volume"
  | "copy"
  | "trash"
  | "delete"
  | "space"
  | "clock"
  | "check"
  | "close"
  | "search"
  | "spark"
  | "shield"
  | "expand"
  | "help"
  | "download"
  | "menu";
export function Icon({
  name,
  size = 20,
  ...props
}: {
  name: IconName;
  size?: number;
  className?: string;
}) {
  const paths: Record<IconName, ReactNode> = {
    hand: (
      <>
        <path d="M8 12V5a1.5 1.5 0 0 1 3 0v6-8a1.5 1.5 0 0 1 3 0v8-6a1.5 1.5 0 0 1 3 0v7-4a1.5 1.5 0 0 1 3 0v7c0 5-3 7-7 7-3 0-5-2-7-5l-3-4a1.7 1.7 0 0 1 2.5-2L8 14" />
      </>
    ),
    scan: (
      <>
        <path d="M8 3H4a1 1 0 0 0-1 1v4m13-5h4a1 1 0 0 1 1 1v4M3 16v4a1 1 0 0 0 1 1h4m8 0h4a1 1 0 0 0 1-1v-4M7 12h10" />
        <path d="m10 9-3 3 3 3m4-6 3 3-3 3" />
      </>
    ),
    book: (
      <>
        <path d="M12 5c-3-2-6-2-10-1v15c4-1 7-1 10 1 3-2 6-2 10-1V4c-4-1-7-1-10 1v15" />
      </>
    ),
    home: (
      <>
        <path d="m3 10 9-7 9 7v10H3zM9 20v-7h6v7" />
      </>
    ),
    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v6m0-10v.1" />
      </>
    ),
    moon: <path d="M20.8 13A9 9 0 0 1 11 3.2 9 9 0 1 0 20.8 13Z" />,
    sun: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" />
      </>
    ),
    arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
    camera: (
      <>
        <path d="m8 5 2-2h4l2 2h4a2 2 0 0 1 2 2v12H2V7a2 2 0 0 1 2-2z" />
        <circle cx="12" cy="12" r="4" />
      </>
    ),
    pause: (
      <>
        <path d="M8 5v14M16 5v14" strokeWidth="3" />
      </>
    ),
    play: <path d="m8 4 12 8-12 8z" />,
    mirror: (
      <>
        <path d="M12 2v20M8 6l-5 12h5zm8 0 5 12h-5z" />
      </>
    ),
    chevron: <path d="m6 9 6 6 6-6" />,
    settings: (
      <>
        <path d="m9 3-1 3-3 1-2 4 2 2v4l4 2 2-1 3 1 4-2v-4l2-2-2-4-3-1-1-3z" />
        <circle cx="11.5" cy="11.5" r="3" />
      </>
    ),
    volume: (
      <>
        <path d="M11 4 5 9H2v6h3l6 5zM15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14" />
      </>
    ),
    copy: (
      <>
        <rect x="8" y="8" width="12" height="13" rx="2" />
        <path d="M16 8V3H3v13h5" />
      </>
    ),
    trash: (
      <>
        <path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7" />
      </>
    ),
    delete: (
      <>
        <path d="M9 5h12v14H9l-7-7zM12 9l6 6m0-6-6 6" />
      </>
    ),
    space: <path d="M4 9v6h16V9M9 19h6" />,
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 6v6l4 2" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    search: (
      <>
        <circle cx="10.5" cy="10.5" r="7" />
        <path d="m16 16 5 5" />
      </>
    ),
    spark: (
      <>
        <path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5zM20 2v4m-2-2h4" />
      </>
    ),
    shield: (
      <>
        <path d="m12 2 9 4v6c0 5-9 10-9 10S3 17 3 12V6z" />
        <path d="m8 12 3 3 5-6" />
      </>
    ),
    expand: <path d="M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5" />,
    help: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M9 8a3 3 0 1 1 4 3c-1 .5-1 1-1 3m0 3v.1" />
      </>
    ),
    download: <path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" />,
    menu: <path d="M4 6h16M4 12h16M4 18h16" />,
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {paths[name]}
    </svg>
  );
}
