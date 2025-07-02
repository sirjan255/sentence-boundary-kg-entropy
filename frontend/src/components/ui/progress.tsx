import React from "react";

export function Progress({ value = 0, className }: { value?: number; className?: string }) {
  return (
    <div style={{ background: "#f1f5f9", borderRadius: 4, height: 8, width: "100%", margin: "8px 0" }}>
      <div
        style={{
          height: "100%",
          borderRadius: 4,
          background: "linear-gradient(90deg, #06b6d4 0%, #818cf8 100%)",
          width: `${Math.max(0, Math.min(100, value))}%`,
          transition: "width 0.4s cubic-bezier(0.4,0,0.2,1)"
        }}
      />
    </div>
  );
}