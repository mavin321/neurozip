export default function TextArea({ value, onChange, placeholder }) {
  return (
    <textarea
      style={{
        width: "100%",
        height: "260px",
        padding: "14px",
        resize: "vertical",
        fontSize: "15px",
        borderRadius: "8px",
        border: "1px solid #d0d7de",
        outline: "none",
      }}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
    />
  );
}
