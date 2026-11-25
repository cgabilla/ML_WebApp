(function () {
  const form = document.getElementById("predict-form");
  if (!form) return;

  const schemaEl = document.getElementById("feature-schema");
  const schema = schemaEl ? JSON.parse(schemaEl.textContent) : [];

  function pick(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

  function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  // Mild bias toward middle values so random students feel realistic,
  // not always extreme "perfect" or "disaster" profiles.
  function randomSoftInt(min, max) {
    const a = randomInt(min, max);
    const b = randomInt(min, max);
    return Math.round((a + b) / 2);
  }

  function buildRandomProfile() {
    const values = {};
    schema.forEach(function (field) {
      if (field.type === "categorical") {
        values[field.name] = String(pick(field.choices).value);
      } else {
        // Mix soft-middle and full-range draws so outcomes stay less obvious.
        values[field.name] =
          Math.random() < 0.65
            ? randomSoftInt(field.min, field.max)
            : randomInt(field.min, field.max);
      }
    });
    return values;
  }

  function applyValues(values) {
    Object.keys(values).forEach(function (name) {
      const el = form.elements.namedItem(name);
      if (!el) return;
      el.value = values[name];
    });
  }

  const surpriseBtn = document.getElementById("surprise-btn");
  if (surpriseBtn) {
    surpriseBtn.addEventListener("click", function () {
      applyValues(buildRandomProfile());
      const first = form.querySelector("select, input");
      if (first) first.focus();
    });
  }

  form.addEventListener("submit", function (event) {
    const submitter = event.submitter;
    if (
      submitter &&
      submitter.getAttribute("formaction") &&
      submitter.getAttribute("formaction").includes("reset")
    ) {
      return;
    }

    const missing = [];
    form.querySelectorAll("[required]").forEach(function (el) {
      if (!String(el.value || "").trim()) {
        const label = form.querySelector('label[for="' + el.id + '"]');
        missing.push(label ? label.childNodes[0].textContent.trim() : el.name);
      }
    });

    if (missing.length) {
      event.preventDefault();
      alert(
        "Some fields are still empty. Please fill in the missing data before predicting.\n\n" +
          missing.slice(0, 8).join(", ") +
          (missing.length > 8 ? ", …" : "")
      );
      for (const el of form.querySelectorAll("[required]")) {
        if (!String(el.value || "").trim()) {
          el.focus();
          break;
        }
      }
    }
  });

  const result = document.getElementById("result");
  if (result) {
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  }
})();
