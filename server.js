import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { InferenceClient } from "@huggingface/inference";
import sharp from "sharp";
import fs from "fs";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

const PORT = process.env.PORT || 10000;
const LOGO_PATH = "./metaweb-logo.jpg";

function getHF() {
  if (!process.env.HF_TOKEN) {
    throw new Error("HF_TOKEN is not configured.");
  }
  return new InferenceClient(process.env.HF_TOKEN);
}

async function addMetaWebLogo(imageBuffer) {
  if (!fs.existsSync(LOGO_PATH)) {
    console.warn("MetaWeb logo not found.");
    return imageBuffer;
  }

  const image = sharp(imageBuffer);
  const meta = await image.metadata();

  const width = meta.width || 1024;
  const height = meta.height || 1024;

  const logoWidth = Math.max(120, Math.round(width * 0.18));
  const margin = Math.max(18, Math.round(width * 0.025));

  const logo = await sharp(LOGO_PATH)
    .resize({
      width: logoWidth,
      fit: "inside"
    })
    .png()
    .toBuffer();

  const logoMeta = await sharp(logo).metadata();
  const logoW = logoMeta.width || logoWidth;
  const logoH = logoMeta.height || 100;

  return await image
    .composite([
      {
        input: logo,
        left: width - logoW - margin,
        top: height - logoH - margin
      }
    ])
    .png()
    .toBuffer();
}

app.get("/", (req, res) => {
  res.json({
    service: "MetaWeb AI Auto Post",
    status: "online"
  });
});

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    service: "metaweb-ai-auto-post",
    hfConfigured: Boolean(process.env.HF_TOKEN),
    logoConfigured: fs.existsSync(LOGO_PATH)
  });
});

app.post("/api/ai/text", async (req, res) => {
  try {
    const topic = String(
      req.body?.topic || "website development"
    );

    const hf = getHF();

    const completion = await hf.chatCompletion({
      model:
        process.env.HF_TEXT_MODEL ||
        "openai/gpt-oss-120b",

      messages: [
        {
          role: "system",
          content:
            "You create professional Facebook posts for MetaWeb, an Indian technology company offering websites, custom software, ERP, Android apps, e-commerce, business automation and AI solutions. Write naturally in Hindi/Hinglish. Do not invent prices, results or guarantees."
        },
        {
          role: "user",
          content:
            `Create a fresh Facebook post about ${topic}. Include headline, useful points, CTA and 5-8 relevant hashtags. Avoid duplicate filler.`
        }
      ],

      temperature: 0.8,
      max_tokens: 700
    });

    const text =
      completion?.choices?.[0]?.message?.content || "";

    res.json({
      ok: true,
      text
    });

  } catch (error) {
    console.error(error);

    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

app.post("/api/ai/image", async (req, res) => {
  try {
    const prompt = String(
      req.body?.prompt ||
      "Professional premium social media advertisement for MetaWeb, technology company, website development, custom software, ERP, Android apps and business automation, modern Indian business style, clean professional technology background, square composition, no readable text"
    );

    const hf = getHF();

    const imageBlob = await hf.textToImage({
      model:
        process.env.HF_IMAGE_MODEL ||
        "black-forest-labs/FLUX.1-dev",

      inputs: prompt
    });

    const originalBuffer =
      Buffer.from(await imageBlob.arrayBuffer());

    const finalBuffer =
      await addMetaWebLogo(originalBuffer);

    const image =
      `data:image/png;base64,${finalBuffer.toString("base64")}`;

    res.json({
      ok: true,
      image,
      logoAdded: fs.existsSync(LOGO_PATH)
    });

  } catch (error) {
    console.error(error);

    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

app.post("/api/ai/generate", async (req, res) => {
  try {
    const topic = String(
      req.body?.topic || "Business automation"
    );

    const hf = getHF();

    const [completion, imageBlob] =
      await Promise.all([

        hf.chatCompletion({
          model:
            process.env.HF_TEXT_MODEL ||
            "openai/gpt-oss-120b",

          messages: [
            {
              role: "system",
              content:
                "You create professional Facebook content for MetaWeb, an Indian technology company offering websites, custom software, ERP, Android apps, e-commerce, business automation and AI solutions. Write naturally in Hindi/Hinglish."
            },
            {
              role: "user",
              content:
                `Create a fresh Facebook post about ${topic}. Include headline, useful explanation, CTA and 5-8 relevant hashtags.`
            }
          ],

          temperature: 0.8,
          max_tokens: 700
        }),

        hf.textToImage({
          model:
            process.env.HF_IMAGE_MODEL ||
            "black-forest-labs/FLUX.1-dev",

          inputs:
            `Professional social media image for MetaWeb about ${topic}. Modern business technology aesthetic, premium, realistic, square format, no readable text.`
        })
      ]);

    const text =
      completion?.choices?.[0]?.message?.content || "";

    const originalBuffer =
      Buffer.from(await imageBlob.arrayBuffer());

    const finalBuffer =
      await addMetaWebLogo(originalBuffer);

    const image =
      `data:image/png;base64,${finalBuffer.toString("base64")}`;

    res.json({
      ok: true,
      topic,
      text,
      image,
      logoAdded: fs.existsSync(LOGO_PATH)
    });

  } catch (error) {
    console.error(error);

    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

app.post("/api/facebook/publish", async (req, res) => {
  res.status(501).json({
    ok: false,
    message:
      "Facebook publishing is not configured yet."
  });
});

app.listen(PORT, () => {
  console.log(
    `MetaWeb AI Auto Post running on port ${PORT}`
  );
});
