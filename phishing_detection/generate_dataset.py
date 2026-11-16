"""
Generate a synthetic (custom-sample) phishing / legitimate email dataset.

The PDF allows "Kaggle, UCI, or custom samples". Because this environment has no
internet access, we synthesize realistic emails from templates, with deliberate
noise (hard negatives such as legit security alerts, and stealthy phishing without
urgency words) so that models do not trivially score 100%.

To use a REAL dataset instead, skip this script and run:
    python run_pipeline.py --data path/to/your.csv --text-col body --label-col label
"""
import random
import pandas as pd

random.seed(42)

BRANDS = ["PayPal", "Amazon", "Microsoft", "Netflix", "SBI Bank", "HDFC Bank", "Apple", "Google", "Facebook", "DHL"]
GOOD_DOMAINS = ["company.com", "university.edu", "gmail.com", "outlook.com", "yahoo.com", "mail.org", "startup.io", "iict.edu.in"]
BAD_TLDS = ["xyz", "top", "info", "click", "ru", "tk", "biz"]
NAMES = ["Rahul", "Priya", "Amit", "Sneha", "John", "Maria", "Ankit", "Neha", "David", "Sara", "Vikram", "Anita"]
SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl"]

PHISH_URGENT = [
    "Dear Customer, your {b} account has been suspended due to suspicious activity. Verify your identity immediately: {url}",
    "URGENT: Unusual sign-in detected on your {b} account. Confirm your password within 24 hours or your account will be closed. {url}",
    "Congratulations! You have won a $1000 {b} gift card. Claim your prize now: {url} Offer expires today!",
    "Your {b} payment failed. Update your billing information immediately to avoid service termination: {url}",
    "Security alert: your {b} password expires today. Click here to keep your account active: {url}",
    "Action required: we could not verify your {b} account details. Login now and confirm your card number: {url}",
    "Final notice! Your package from {b} is on hold. Pay the small customs fee here: {url}",
    "Dear user, a refund of Rs. 4,999 is pending in your {b} account. Submit bank details to receive it: {url}",
]
# stealthy phishing: polite, business-like, little urgency
PHISH_STEALTH = [
    "Hi {n}, please review the shared document regarding the Q3 invoice. Sign in to view: {url}",
    "Hello, {n} shared a file with you. Open the attachment portal to access it: {url}",
    "Your {b} statement is ready. To view it please log in here: {url}",
    "Hi, can you check this updated payroll schedule? Access it through our new portal: {url}",
    "Voicemail received from {n}. Listen to the message using the secure link: {url}",
]
LEGIT_TEMPLATES = [
    "Hi {n}, attached is the agenda for tomorrow's project meeting at 10 AM. Let me know if you want to add anything.",
    "Hello team, the quarterly report has been uploaded to the shared drive. Please review it before Friday.",
    "Hi {n}, thanks for your help yesterday. Lunch on Thursday? Let me know what time works for you.",
    "Dear students, the assignment deadline has been extended to next Monday. Submit it through the university portal.",
    "Your {b} order #{num} has shipped and will arrive on Thursday. Track it in the {b} app.",
    "Reminder: your dentist appointment is scheduled for Wednesday at 4 PM. Reply to this email to reschedule.",
    "Hi {n}, here are the notes from our call. Next steps: finalize the design, schedule the review, update the sheet.",
    "Thank you for your purchase from {b}. Your receipt is attached. No action is needed.",
    "Newsletter: this week in tech - new Python release, conference announcements and open source highlights.",
    "Hi {n}, can you send me the slides from the presentation? I want to go through them before the interview.",
]
# hard negatives: legit mails that sound urgent / contain links
LEGIT_HARD = [
    "Security notice from {b}: a new device signed in to your account. If this was you, no action is needed. Review activity at https://www.{d}/security",
    "Urgent: server maintenance tonight at 11 PM. Please save your work before then. Details: https://wiki.{d}/maintenance",
    "Your {b} password was changed successfully. If you did not make this change, contact support through our official website https://www.{d}/help",
    "Reminder: your invoice #{num} is due on the 30th. You can pay from your dashboard at https://billing.{d}",
    "Hi {n}, the deadline is today! Please push your code and open a pull request: https://github.com/team/repo/pulls",
]

def rand_good_url():
    d = random.choice(GOOD_DOMAINS)
    return f"https://www.{d}/" + random.choice(["login", "account", "docs", "portal", "help"])

def rand_bad_url(brand):
    kind = random.random()
    b = brand.lower().replace(" ", "")
    if kind < 0.35:
        return f"http://{b}-secure-login.{random.choice(BAD_TLDS)}/verify?id={random.randint(1000,99999)}"
    if kind < 0.6:
        return f"http://{random.randint(11,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}/{b}/signin"
    if kind < 0.8:
        return f"https://{random.choice(SHORTENERS)}/{random.choice('abcdefghij')}{random.randint(100,999)}"
    return f"http://{b}.account-update.{random.choice(BAD_TLDS)}/login@confirm"

FILLER = ["Thanks and regards.", "Please let me know if you have questions.", "Best wishes, {n}.", "Sent from my phone.",
          "This message was sent to you as a registered user.", "Do not share this email.", "Have a great week!", "See details below.",
          "Please find the information attached.", "Kindly review at your convenience.", "Your reference number is {num}."]

def add_filler(text):
    extra = " ".join(random.sample(FILLER, random.randint(0, 3)))
    return (text + " " + extra).format(n=random.choice(NAMES), num=random.randint(100000, 999999)) if "{" in extra else text + " " + extra

def maybe_html(text):
    text = add_filler(text)
    if random.random() < 0.3:
        return f"<html><body><p>{text}</p><br><a href='#'>click</a></body></html>"
    return text

def make_phish():
    b, n = random.choice(BRANDS), random.choice(NAMES)
    tpl = random.choice(PHISH_STEALTH if random.random() < 0.45 else PHISH_URGENT)
    url = rand_bad_url(b)
    # a few phishing mails use a clean-looking url (noise)
    if random.random() < 0.25:
        url = rand_good_url()
    body = tpl.format(b=b, n=n, url=url)
    if random.random() < 0.3:
        body = body.upper() if random.random() < 0.2 else body + " !!!"
    sender_domain = f"{b.lower().replace(' ', '')}-support.{random.choice(BAD_TLDS)}" if random.random() < 0.6 else random.choice(GOOD_DOMAINS)
    subject = random.choice(["Account suspended", "Verify your account", "You won!", "Payment failed", "Security alert", "Re: invoice", "Shared document", "Voicemail"])
    return subject, sender_domain, maybe_html(body), 1

def make_legit():
    b, n, d = random.choice(BRANDS), random.choice(NAMES), random.choice(GOOD_DOMAINS)
    if random.random() < 0.35:
        tpl = random.choice(LEGIT_HARD)
    else:
        tpl = random.choice(LEGIT_TEMPLATES)
    body = tpl.format(b=b, n=n, d=d, num=random.randint(10000, 99999))
    if random.random() < 0.3:
        body += " " + rand_good_url()
    sender_domain = random.choice(GOOD_DOMAINS) if random.random() < 0.85 else f"{b.lower().replace(' ', '')}-support.{random.choice(BAD_TLDS)}"
    subject = random.choice(["Meeting agenda", "Lunch?", "Assignment update", "Your order has shipped", "Reminder", "Notes", "Receipt", "Weekly newsletter", "Security notice", "Maintenance"])
    return subject, sender_domain, maybe_html(body), 0

def main(n=3000, path="data/raw_emails.csv"):
    rows = [make_phish() if i % 2 == 0 else make_legit() for i in range(n)]
    random.shuffle(rows)
    # ~3% label noise simulates annotation errors found in real datasets
    rows = [(a, b, c, 1 - d if random.random() < 0.03 else d) for a, b, c, d in rows]
    df = pd.DataFrame(rows, columns=["subject", "sender_domain", "body", "label"])
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} emails to {path}  (phishing={df.label.sum()}, legitimate={(df.label==0).sum()})")

if __name__ == "__main__":
    main()
