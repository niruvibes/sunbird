// @description Shows how long you've been in a voice channel or call
// @version 1.1.0

const setCallTimer = phoenix.ui.setCallTimer;

if (setCallTimer) {
  const settings = phoenix.settings.define({
    format: {
      type: "select",
      title: "Timer format",
      options: [
        { value: "stopwatch", label: "1d 02:03:04" },
        { value: "human", label: "1d 2h 3m 4s" },
      ],
      default: "stopwatch",
      onChange: (format) => setCallTimer(format === "human" ? "human" : "stopwatch"),
    },
  });

  setCallTimer(settings.format);
}
