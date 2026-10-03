import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { InferenceClient } from "@huggingface/inference";

dotenv.config();
const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));
const PORT = process.env.PORT || 10000;
const textModel = process.env.HF_TEXT_MODEL || "openai/gpt-oss-120b";
const imageModel = process.env.HF_IMAGE_MODEL || "black-forest-labs/FLUX.1-dev";

function hf() {
  if (!process.env.HF_TOKEN) throw new Error("HF_TOKEN is not configured in Render Environment Variables.");
  return new InferenceClient(process.env.HF_TOKEN);
}

app.get("/", (req,res)=>res.json({service:"MetaWeb AI Auto Post",status:"online"}));
app.get("/api/health", (req,res)=>res.json({ok:true,service:"metaweb-ai-auto-post",hfConfigured:Boolean(process.env.HF_TOKEN)}));
app.get("/api/config", (req,res)=>res.json({aiProvider:"huggingface",textModel,imageModel,facebookConfigured:Boolean(process.env.META_PAGE_ID&&process.env.META_PAGE_ACCESS_TOKEN)}));

app.post("/api/ai/text", async (req,res)=>{
  try {
    const topic=String(req.body?.topic||"website development");
    const r=await hf().chatCompletion({model:textModel,messages:[
      {role:"system",content:"You create professional Facebook content for MetaWeb, an Indian technology company offering websites, custom software, ERP, Android apps, e-commerce, business automation and AI solutions. Write in natural Hindi/Hinglish. Do not invent prices, results or guarantees."},
      {role:"user",content:`Create a fresh Facebook post about: ${topic}. Include a useful headline, 3-5 concise points, a short CTA, and 5-8 relevant hashtags.`}
    ],temperature:0.8,max_tokens:700});
    res.json({ok:true,text:r?.choices?.[0]?.message?.content||"",model:textModel});
  } catch(e) { console.error(e); res.status(500).json({ok:false,error:e?.message||"AI text generation failed"}); }
});

app.post("/api/ai/image", async (req,res)=>{
  try {
    const prompt=String(req.body?.prompt||"A premium modern technology business social media graphic for MetaWeb, professional website development and business automation, clean Indian business aesthetic, realistic, square composition, no readable text");
    const blob=await hf().textToImage({model:imageModel,inputs:prompt});
    const buffer=Buffer.from(await blob.arrayBuffer());
    res.json({ok:true,image:`data:${blob.type||"image/png"};base64,${buffer.toString("base64")}`,model:imageModel});
  } catch(e) { console.error(e); res.status(500).json({ok:false,error:e?.message||"AI image generation failed"}); }
});

app.post("/api/ai/generate", async (req,res)=>{
  try {
    const topic=String(req.body?.topic||"Business automation");
    const prompt=String(req.body?.imagePrompt||`Professional social media image for MetaWeb about ${topic}. Modern business technology aesthetic, clean composition, realistic, premium, square format, no readable text.`);
    const client=hf();
    const [r,blob]=await Promise.all([
      client.chatCompletion({model:textModel,messages:[
        {role:"system",content:"You create professional Facebook content for MetaWeb, an Indian technology company offering websites, custom software, ERP, Android apps, e-commerce, business automation and AI solutions. Write in natural Hindi/Hinglish. Do not invent prices, results or guarantees."},
        {role:"user",content:`Create a fresh Facebook post about \"${topic}\". Include a headline, useful explanation, CTA and 5-8 relevant hashtags.`}
      ],temperature:0.8,max_tokens:700}),
      client.textToImage({model:imageModel,inputs:prompt})
    ]);
    const buffer=Buffer.from(await blob.arrayBuffer());
    res.json({ok:true,topic,text:r?.choices?.[0]?.message?.content||"",image:`data:${blob.type||"image/png"};base64,${buffer.toString("base64")}`,textModel,imageModel});
  } catch(e) { console.error(e); res.status(500).json({ok:false,error:e?.message||"AI generation failed"}); }
});

app.post("/api/facebook/publish", (req,res)=>res.status(501).json({ok:false,message:"Facebook publishing is not configured yet."}));
app.listen(PORT,()=>console.log(`MetaWeb AI Auto Post server running on port ${PORT}`));
